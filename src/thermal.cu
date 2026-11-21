#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
void check(cudaError_t error)
{
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
template <class T>
struct Device
{
    T *p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device()
    {
        if (p)
            cudaFree(p);
    }
    Device(const Device &) = delete;
    void zero() { check(cudaMemset(p, 0, n * sizeof(T))); }
    void upload(const std::vector<T> &v) { check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> download() const
    {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
template <class T>
std::vector<T> read_array(std::ifstream &f, size_t n)
{
    std::vector<T> v(n);
    if (!f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)))
        throw std::runtime_error("Truncated thermal input");
    return v;
}
struct Input
{
    int n, nt, nf, nr, nnz;
    std::vector<double> xyz, k, h, eps, pcb, rho_cp;
    std::vector<int> tet, region, face, tslot, fslot, row, col;
    explicit Input(const fs::path &path)
    {
        std::ifstream f(path, std::ios::binary);
        char magic[8];
        if (!f.read(magic, 8) || (std::string(magic, 8) != "POLTHM01" && std::string(magic,8) != "POLTHM02"))
            throw std::runtime_error("Invalid thermal input header");
        auto counts = read_array<int>(f, 5);
        n = counts[0];
        nt = counts[1];
        nf = counts[2];
        nr = counts[3];
        nnz = counts[4];
        if (std::min({n, nt, nf, nr, nnz}) <= 0)
            throw std::runtime_error("Invalid thermal mesh counts");
        xyz = read_array<double>(f, size_t(n) * 3);
        tet = read_array<int>(f, size_t(nt) * 4);
        region = read_array<int>(f, nt);
        k = read_array<double>(f, nt);
        face = read_array<int>(f, size_t(nf) * 3);
        h = read_array<double>(f, nf);
        eps = read_array<double>(f, nf);
        pcb = read_array<double>(f, nf);
        tslot = read_array<int>(f, size_t(nt) * 16);
        fslot = read_array<int>(f, size_t(nf) * 9);
        row = read_array<int>(f, n + 1);
        col = read_array<int>(f, nnz);
        rho_cp = std::string(magic,8)=="POLTHM02" ? read_array<double>(f,nt) : std::vector<double>(nt,0);
        for (int i : tet)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid tetrahedron node");
        for (int i : face)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid boundary node");
        for (int i : region)
            if (i <= 0 || i >= nr)
                throw std::runtime_error("Invalid material region");
        for (int i : tslot)
            if (i < 0 || i >= nnz)
                throw std::runtime_error("Invalid element CSR slot");
        for (int i : fslot)
            if (i < 0 || i >= nnz)
                throw std::runtime_error("Invalid face CSR slot");
        for (int i : col)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid CSR column");
        if (row.front() != 0 || row.back() != nnz || !std::is_sorted(row.begin(), row.end()))
            throw std::runtime_error("Invalid CSR offsets");
    }
};
struct Physics
{
    double vin, vout, iout, fsw, L, rhs, rls, tref, alpha, tr, tf, dead, transitions, vdiode, dcr, cu_alpha, ac, core, ambient, trad;
    int hs, ls, winding, magnetic, nr;
};
std::map<std::string, double> parameters(const fs::path &p)
{
    std::ifstream f(p);
    std::map<std::string, double> m;
    std::string key;
    double value;
    while (f >> key >> value)
    {
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite parameter");
        m[key] = value;
    }
    if (!f.eof())
        throw std::runtime_error("Invalid parameter file");
    return m;
}
__device__ double3 point(const double *xyz, int i) { return make_double3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]); }
__device__ double3 sub(double3 a, double3 b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ double3 scale(double3 a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
__device__ double3 cross3(double3 a, double3 b) { return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ double dot3(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ void maximum(double *address, double value)
{
    auto p = reinterpret_cast<unsigned long long *>(address);
    auto old = *p, assumed = old;
    do
    {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(fmax(value, __longlong_as_double(assumed))));
    } while (old != assumed);
}
__global__ void stiffness(int nt, const int *tet, const int *region, const double *xyz, const double *k, const int *slots,
                          double *matrix, double *volume, double *region_volume, int *invalid)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nt)
        return;
    double3 a = point(xyz, tet[4 * e]), u = sub(point(xyz, tet[4 * e + 1]), a), v = sub(point(xyz, tet[4 * e + 2]), a), w = sub(point(xyz, tet[4 * e + 3]), a);
    double det = dot3(u, cross3(v, w));
    if (!(det > 0) || !(k[e] > 0))
    {
        atomicExch(invalid, 1);
        return;
    }
    double3 g[4];
    g[1] = scale(cross3(v, w), 1 / det);
    g[2] = scale(cross3(w, u), 1 / det);
    g[3] = scale(cross3(u, v), 1 / det);
    g[0] = make_double3(-g[1].x - g[2].x - g[3].x, -g[1].y - g[2].y - g[3].y, -g[1].z - g[2].z - g[3].z);
    volume[e] = det / 6;
    atomicAdd(region_volume + region[e], volume[e]);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            atomicAdd(matrix + slots[16 * e + 4 * i + j], k[e] * volume[e] * dot3(g[i], g[j]));
}
__global__ void initialize_stats(int nr, double *integral, double *max_t, double *neg_min)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr)
        return;
    integral[r] = 0;
    max_t[r] = -1e300;
    neg_min[r] = -1e300;
}
__global__ void statistics(int nt, const int *tet, const int *region, const double *volume, const double *theta, double ambient,
                           double *integral, double *max_t, double *neg_min)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nt)
        return;
    double avg = 0;
    int r = region[e];
    for (int i = 0; i < 4; ++i)
    {
        double t = theta[tet[4 * e + i]] + ambient;
        avg += t / 4;
        maximum(max_t + r, t);
        maximum(neg_min + r, -t);
    }
    atomicAdd(integral + r, avg * volume[e]);
}
__global__ void mean_temperatures(int nr, const double *integral, const double *volume, double ambient, double *mean)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr)
        mean[r] = volume[r] > 0 ? integral[r] / volume[r] : ambient;
}
// Order: HS conduction, HS switching, LS conduction, dead-time, DC copper, AC copper, core, auxiliary, total.
__global__ void losses(Physics c, const double *mean, double *power, double *components, int *invalid)
{
    if (threadIdx.x || blockIdx.x)
        return;
    for (int i = 0; i < c.nr; ++i)
        power[i] = 0;
    double duty = c.vout / c.vin, ripple = c.vout * (1 - duty) / (c.L * c.fsw), irms2 = c.iout * c.iout + ripple * ripple / 12;
    double rh = c.rhs * (1 + c.alpha * (mean[c.hs] - c.tref)), rl = c.rls * (1 + c.alpha * (mean[c.ls] - c.tref));
    double rd = c.dcr * (1 + c.cu_alpha * (mean[c.winding] - c.tref));
    if (rh < 0 || rl < 0 || rd < 0 || !isfinite(rh) || !isfinite(rl) || !isfinite(rd))
    {
        *invalid = 1;
        return;
    }
    components[0] = irms2 * rh * duty;
    components[1] = .5 * c.vin * c.iout * (c.tr + c.tf) * c.fsw;
    components[2] = irms2 * rl * (1 - duty);
    components[3] = c.vdiode * c.iout * c.dead * c.transitions * c.fsw;
    components[4] = irms2 * rd;
    components[5] = c.ac;
    components[6] = c.core;
    components[7] = 0;
    double total = 0;
    for (int i = 0; i < 8; ++i)
        total += components[i];
    components[8] = total;
    power[c.hs] = components[0] + components[1];
    power[c.ls] = components[2] + components[3];
    power[c.winding] = components[4] + components[5];
    power[c.magnetic] = components[6];
}
__global__ void source(int nt, const int *tet, const int *region, const double *volume, const double *region_volume, const double *power, double *rhs)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nt)
        return;
    int r = region[e];
    double load = power[r] * volume[e] / region_volume[r] / 4;
    for (int i = 0; i < 4; ++i)
        atomicAdd(rhs + tet[4 * e + i], load);
}
__global__ void integrated_sources(int nt, const int *region, const double *volume, const double *rv, const double *power, double *integrated)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < nt)
        atomicAdd(integrated + region[e], power[region[e]] * volume[e] / rv[region[e]]);
}
__device__ double face_area(const double *xyz, const int *face, int e)
{
    double3 a = point(xyz, face[3 * e]);
    double3 c = cross3(sub(point(xyz, face[3 * e + 1]), a), sub(point(xyz, face[3 * e + 2]), a));
    return .5 * sqrt(dot3(c, c));
}
// Three-point degree-two triangle quadrature: each weight is A/3.
__device__ double shape(int q, int i) { return q == i ? 2.0 / 3 : 1.0 / 6; }
__global__ void boundary_matrix(int nf, const int *face, const int *slots, const double *xyz, const double *h, const double *eps, const double *pcb,
                                const double *theta, double ambient, double trad, double *matrix, double *rhs)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nf)
        return;
    double area = face_area(xyz, face, e), sigma = 5.670374419e-8;
    for (int q = 0; q < 3; ++q)
    {
        double t = ambient;
        for (int i = 0; i < 3; ++i)
            t += shape(q, i) * theta[face[3 * e + i]];
        double hr = eps[e] * sigma * (t + trad) * (t * t + trad * trad);
        for (int i = 0; i < 3; ++i)
        {
            atomicAdd(rhs + face[3 * e + i], area / 3 * shape(q, i) * hr * (trad - ambient));
            for (int j = 0; j < 3; ++j)
                atomicAdd(matrix + slots[9 * e + 3 * i + j], area / 3 * shape(q, i) * shape(q, j) * (h[e] + pcb[e] + hr));
        }
    }
}
__global__ void heat_flow(int nf, const int *face, const double *xyz, const double *h, const double *eps, const double *pcb,
                          const double *theta, double ambient, double trad, double *flow)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nf)
        return;
    double area = face_area(xyz, face, e), conv = 0, rad = 0, board = 0;
    for (int q = 0; q < 3; ++q)
    {
        double rise = 0;
        for (int i = 0; i < 3; ++i)
            rise += shape(q, i) * theta[face[3 * e + i]];
        double t = ambient + rise;
        conv += area / 3 * h[e] * rise;
        board += area / 3 * pcb[e] * rise;
        rad += area / 3 * eps[e] * 5.670374419e-8 * (t * t * t * t - trad * trad * trad * trad);
    }
    atomicAdd(flow, conv);
    atomicAdd(flow + 1, rad);
    atomicAdd(flow + 2, board);
}
__global__ void spmv(int n, const int *row, const int *col, const double *a, const double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double s = 0;
    for (int j = row[i]; j < row[i + 1]; ++j)
        s += a[j] * x[col[j]];
    y[i] = s;
}
__global__ void residual(int n, const double *b, const double *ax, double *r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        r[i] = b[i] - ax[i];
}
__global__ void diagonal(int n, const int *row, const int *col, const double *a, double *diag, int *invalid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double d = 0;
    for (int j = row[i]; j < row[i + 1]; ++j)
        if (col[j] == i)
            d = a[j];
    if (!(d > 0) || !isfinite(d))
        atomicExch(invalid, 1);
    diag[i] = d;
}
__global__ void precondition(int n, const double *r, const double *diag, double *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = r[i] / diag[i];
}
__global__ void update_xr(int n, double alpha, const double *p, const double *ap, double *x, double *r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
    }
}
__global__ void direction(int n, double beta, const double *z, double *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = z[i] + beta * p[i];
}
__global__ void dot_product(int n, const double *a, const double *b, double *result)
{
    __shared__ double values[256];
    double s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        s += a[i] * b[i];
    values[threadIdx.x] = s;
    __syncthreads();
    for (int d = 128; d; d /= 2)
    {
        if (threadIdx.x < d)
            values[threadIdx.x] += values[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(result, values[0]);
}
__global__ void relax(int n, const double *candidate, double omega, double ambient, double limit, double *theta, double *delta, int *invalid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double change = omega * (candidate[i] - theta[i]);
    theta[i] += change;
    maximum(delta, fabs(change));
    if (!isfinite(theta[i]) || theta[i] + ambient <= 0 || theta[i] + ambient > limit)
        atomicExch(invalid, 1);
}
__global__ void absolute_temperature(int n, const double *theta, double ambient, double *t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        t[i] = theta[i] + ambient;
}
__global__ void lumped_capacity(int nt,const int* tet,const double* volume,const double* rho_cp,double* capacity) {
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=nt)return;
    for(int i=0;i<4;++i)atomicAdd(capacity+tet[4*e+i],rho_cp[e]*volume[e]/4);
}
__global__ void initial_field(int n,double rise,double* theta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)theta[i]=rise;
}
__global__ void time_matrix(int n,const int* row,const int* col,const double* capacity,const double* old,double dt,double* a,double* rhs) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double coefficient=capacity[i]/dt;
    for(int j=row[i];j<row[i+1];++j)if(col[j]==i)a[j]+=coefficient;
    rhs[i]+=coefficient*old[i];
}
__global__ void stored_heat_rate(int n,const double* capacity,const double* theta,const double* old,double dt,double* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(result,capacity[i]*(theta[i]-old[i])/dt);
}
int blocks(int n) { return (n + 255) / 256; }
struct PCGResult
{
    int iterations;
    double relative_residual;
    bool converged;
};
struct PCG
{
    int n;
    Device<double> r, z, p, ap, diag, sum;
    Device<int> invalid;
    explicit PCG(int count) : n(count), r(n), z(n), p(n), ap(n), diag(n), sum(1), invalid(1) {}
    double dot(const double *a, const double *b)
    {
        sum.zero();
        dot_product<<<std::min(blocks(n), 256), 256>>>(n, a, b, sum.p);
        return sum.download()[0];
    }
    PCGResult solve(const int *row, const int *col, const double *a, const double *rhs, double *x, int max_iterations, double rel, double abs)
    {
        invalid.zero();
        diagonal<<<blocks(n), 256>>>(n, row, col, a, diag.p, invalid.p);
        if (invalid.download()[0])
            throw std::runtime_error("Nonpositive or invalid FEM diagonal");
        spmv<<<blocks(n), 256>>>(n, row, col, a, x, ap.p);
        residual<<<blocks(n), 256>>>(n, rhs, ap.p, r.p);
        double bnorm = sqrt(dot(rhs, rhs)), threshold = std::max(abs, rel * bnorm), rn = sqrt(dot(r.p, r.p));
        if (rn <= threshold)
            return {0, rn / std::max(bnorm, 1e-300), true};
        precondition<<<blocks(n), 256>>>(n, r.p, diag.p, z.p);
        check(cudaMemcpy(p.p, z.p, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double rz = dot(r.p, z.p);
        for (int i = 1; i <= max_iterations; ++i)
        {
            spmv<<<blocks(n), 256>>>(n, row, col, a, p.p, ap.p);
            double pap = dot(p.p, ap.p);
            if (!(pap > 0) || !std::isfinite(pap) || !(rz > 0))
                throw std::runtime_error("CUDA PCG breakdown: operator/roundoff requires investigation");
            update_xr<<<blocks(n), 256>>>(n, rz / pap, p.p, ap.p, x, r.p);
            rn = sqrt(dot(r.p, r.p));
            if (rn <= threshold)
            {
                // Explicit residual guards against accumulated recursive-PCG error.
                spmv<<<blocks(n), 256>>>(n, row, col, a, x, ap.p);
                residual<<<blocks(n), 256>>>(n, rhs, ap.p, r.p);
                rn = sqrt(dot(r.p, r.p));
                if (rn <= threshold)
                    return {i, rn / std::max(bnorm, 1e-300), true};
                precondition<<<blocks(n), 256>>>(n, r.p, diag.p, z.p);
                check(cudaMemcpy(p.p, z.p, n * sizeof(double), cudaMemcpyDeviceToDevice));
                rz = dot(r.p, z.p);
                continue;
            }
            precondition<<<blocks(n), 256>>>(n, r.p, diag.p, z.p);
            double next = dot(r.p, z.p);
            direction<<<blocks(n), 256>>>(n, next / rz, z.p, p.p);
            rz = next;
        }
        return {max_iterations, rn / std::max(bnorm, 1e-300), false};
    }
};

// Generic analytic element/linear-algebra checks on GPU, not a CPU solver.
void self_test()
{
    Device<double> xyz(12), k(1), a(16), volume(1), rv(2), theta(4), rhs(4), h(1), eps(1), pcb(1), flow(3), x(4), known(4), power(2);
    Device<int> tet(4), region(1), slots(16), face(3), fslots(9), row(5), col(16), invalid(1);
    xyz.upload({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1});
    k.upload({6});
    tet.upload({0, 1, 2, 3});
    region.upload({1});
    std::vector<int> indices(16);
    for (int i = 0; i < 16; ++i)
        indices[i] = i;
    slots.upload(indices);
    row.upload({0, 4, 8, 12, 16});
    for (int i = 0; i < 16; ++i)
        indices[i] = i % 4;
    col.upload(indices);
    a.zero();
    rv.zero();
    invalid.zero();
    stiffness<<<1, 256>>>(1, tet.p, region.p, xyz.p, k.p, slots.p, a.p, volume.p, rv.p, invalid.p);
    const double expected[] = {3, -1, -1, -1, -1, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1};
    auto matrix = a.download();
    for (int i = 0; i < 16; ++i)
        if (std::abs(matrix[i] - expected[i]) > 1e-13)
            throw std::runtime_error("Self-test: tetrahedral stiffness failed");
    power.upload({0, 2});
    rhs.zero();
    source<<<1, 256>>>(1, tet.p, region.p, volume.p, rv.p, power.p, rhs.p);
    for (double v : rhs.download())
        if (std::abs(v - .5) > 1e-13)
            throw std::runtime_error("Self-test: integrated source failed");
    face.upload({1, 2, 3});
    fslots.upload({5, 6, 7, 9, 10, 11, 13, 14, 15});
    h.upload({2});
    eps.zero();
    pcb.zero();
    theta.zero();
    rhs.zero();
    boundary_matrix<<<1, 256>>>(1, face.p, fslots.p, xyz.p, h.p, eps.p, pcb.p, theta.p, 300, 300, a.p, rhs.p);
    auto boundary = a.download();
    double area = std::sqrt(3.0) / 2;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            double robin = (i && j) ? 2 * area / 12 * (i == j ? 2 : 1) : 0;
            if (std::abs(boundary[4 * i + j] - expected[4 * i + j] - robin) > 1e-13)
                throw std::runtime_error("Self-test: Robin matrix failed");
        }
    known.upload({1, 2, 3, 4});
    spmv<<<1, 256>>>(4, row.p, col.p, a.p, known.p, rhs.p);
    x.zero();
    PCG cg(4);
    auto result = cg.solve(row.p, col.p, a.p, rhs.p, x.p, 50, 1e-12, 1e-14);
    auto solved = x.download();
    for (int i = 0; i < 4; ++i)
        if (!result.converged || std::abs(solved[i] - (i + 1)) > 1e-10)
            throw std::runtime_error("Self-test: GPU PCG failed");
    theta.upload({20, 20, 20, 20});
    eps.upload({.5});
    pcb.upload({3});
    flow.zero();
    heat_flow<<<1, 256>>>(1, face.p, xyz.p, h.p, eps.p, pcb.p, theta.p, 300, 300, flow.p);
    auto out = flow.download();
    double radiation = .5 * 5.670374419e-8 * area * (std::pow(320., 4) - std::pow(300., 4));
    if (std::abs(out[0] - 2 * area * 20) > 1e-10 || std::abs(out[1] - radiation) > 1e-10 || std::abs(out[2] - 3 * area * 20) > 1e-10)
        throw std::runtime_error("Self-test: boundary energy quadrature failed");
    // Adiabatic uniform heating: C_total=1 J/K, P=2 W, dt=.5 s -> +1 K.
    Device<double> rc(1),capacity(4),old(4),stored(1);
    rc.upload({6});capacity.zero();
    lumped_capacity<<<1,256>>>(1,tet.p,volume.p,rc.p,capacity.p);
    for(double c:capacity.download())if(std::abs(c-.25)>1e-13)throw std::runtime_error("Self-test: lumped capacity failed");
    a.zero();rv.zero();stiffness<<<1,256>>>(1,tet.p,region.p,xyz.p,k.p,slots.p,a.p,volume.p,rv.p,invalid.p);
    rhs.zero();source<<<1,256>>>(1,tet.p,region.p,volume.p,rv.p,power.p,rhs.p);
    old.upload({20,20,20,20});x.upload({20,20,20,20});
    time_matrix<<<1,256>>>(4,row.p,col.p,capacity.p,old.p,.5,a.p,rhs.p);
    auto step=cg.solve(row.p,col.p,a.p,rhs.p,x.p,50,1e-12,1e-14);
    for(double t:x.download())if(!step.converged||std::abs(t-21)>1e-10)throw std::runtime_error("Self-test: backward Euler heating failed");
    stored.zero();stored_heat_rate<<<1,256>>>(4,capacity.p,x.p,old.p,.5,stored.p);
    if(std::abs(stored.download()[0]-2)>1e-10)throw std::runtime_error("Self-test: transient energy storage failed");
    check(cudaGetLastError());
    std::cout << "GPU self-tests passed: tetrahedron, sources, Robin matrix, PCG, boundary heat, capacity and transient adiabatic heating.\n";
}

int run(const fs::path &directory)
{
    Input in(directory / "input.bin");
    auto v = parameters(directory / "parameters.txt");
    bool transient=v.count("transient") && v.at("transient")!=0;
    if(transient) for(double rc:in.rho_cp) if(!std::isfinite(rc)||rc<=0)
        throw std::runtime_error("Transient input requires positive volumetric heat capacity");
    auto value = [&](const std::string &key)
    { return v.at(key); };
    Physics c{value("vin"), value("vout"), value("iout"), value("fsw"), value("inductance"), value("rhs_reference"), value("rls_reference"),
              value("reference_k"), value("mosfet_alpha"), value("rise_s"), value("fall_s"), value("dead_s"), value("dead_transitions"), value("diode_v"),
              value("dcr_reference"), value("copper_alpha"), value("ac_loss_w"), value("core_loss_w"), value("ambient_k"), value("radiation_k"),
              int(value("hs_region")), int(value("ls_region")), int(value("winding_region")), int(value("core_region")), in.nr};
    if (value("auxiliary_w") != 0)
        throw std::runtime_error("Control logic is absent; auxiliary power must be zero");
    for (int r : {c.hs, c.ls, c.winding, c.magnetic})
        if (r <= 0 || r >= in.nr)
            throw std::runtime_error("Missing source region");
    Device<int> tet(in.tet.size()), region(in.region.size()), face(in.face.size()), tslot(in.tslot.size()), fslot(in.fslot.size()), row(in.row.size()), col(in.col.size()), invalid(1);
    tet.upload(in.tet);
    region.upload(in.region);
    face.upload(in.face);
    tslot.upload(in.tslot);
    fslot.upload(in.fslot);
    row.upload(in.row);
    col.upload(in.col);
    Device<double> xyz(in.xyz.size()), k(in.k.size()), h(in.h.size()), eps(in.eps.size()), pcb(in.pcb.size());
    xyz.upload(in.xyz);
    k.upload(in.k);
    h.upload(in.h);
    eps.upload(in.eps);
    pcb.upload(in.pcb);
    Device<double> capacity(in.n),rho_cp(in.nt),old_time(in.n),storage(1);
    capacity.zero();rho_cp.upload(in.rho_cp);
    double active_dt=0;
    Device<double> base(in.nnz), a(in.nnz), rhs(in.n), volume(in.nt), rv(in.nr), theta(in.n), candidate(in.n), integral(in.nr), maxt(in.nr), negmin(in.nr), mean(in.nr), power(in.nr), loss(9), flow(3), delta(1), kelvin(in.n);
    base.zero();
    rv.zero();
    theta.zero();
    invalid.zero();
    stiffness<<<blocks(in.nt), 256>>>(in.nt, tet.p, region.p, xyz.p, k.p, tslot.p, base.p, volume.p, rv.p, invalid.p);
    if (invalid.download()[0])
        throw std::runtime_error("Invalid tetrahedron or material conductivity");
    if(transient) {
        lumped_capacity<<<blocks(in.nt),256>>>(in.nt,tet.p,volume.p,rho_cp.p,capacity.p);
        initial_field<<<blocks(in.n),256>>>(in.n,value("initial_temperature_k")-c.ambient,theta.p);
    }
    auto volumes = rv.download();
    for (int r : {c.hs, c.ls, c.winding, c.magnetic})
        if (!(volumes[r] > 0))
            throw std::runtime_error("Zero-volume heat-source region");
    auto stats = [&]()
    {initialize_stats<<<blocks(in.nr),256>>>(in.nr,integral.p,maxt.p,negmin.p);statistics<<<blocks(in.nt),256>>>(in.nt,tet.p,region.p,volume.p,theta.p,c.ambient,integral.p,maxt.p,negmin.p);mean_temperatures<<<blocks(in.nr),256>>>(in.nr,integral.p,rv.p,c.ambient,mean.p); };
    auto update_losses = [&]()
    {losses<<<1,1>>>(c,mean.p,power.p,loss.p,invalid.p);if(invalid.download()[0])throw std::runtime_error("Temperature-dependent resistance outside model domain"); };
    auto assemble = [&]()
    {check(cudaMemcpy(a.p,base.p,in.nnz*sizeof(double),cudaMemcpyDeviceToDevice));rhs.zero();source<<<blocks(in.nt),256>>>(in.nt,tet.p,region.p,volume.p,rv.p,power.p,rhs.p);boundary_matrix<<<blocks(in.nf),256>>>(in.nf,face.p,fslot.p,xyz.p,h.p,eps.p,pcb.p,theta.p,c.ambient,c.trad,a.p,rhs.p);if(active_dt>0)time_matrix<<<blocks(in.n),256>>>(in.n,row.p,col.p,capacity.p,old_time.p,active_dt,a.p,rhs.p);check(cudaGetLastError()); };
    stats();
    update_losses();
    PCG pcg(in.n);
    std::ofstream history(directory / "history.tsv");
    history << std::setprecision(17) << "time_s\tstep\touter\tpcg_iterations\tlinear_relative_residual\tnonlinear_relative_residual\tmax_change_K\tmax_loss_change_W\tpower_W\tconvection_W\tradiation_W\tpcb_W\tstorage_W\tenergy_relative_error\n";
    bool converged=false;
    std::string reason="maximum_outer_iterations";
    int outer=0,step=0,total_outer=0;
    PCGResult linear{0,0,false};
    double nonlinear=0,change=0,loss_change=0,energy=0,stored=0,time=0,last_snapshot=-1;
    auto components=loss.download();std::vector<double> flows(3);
    double accepted_stored=0,accepted_energy=1;
    std::ofstream frame_times,frame_data,time_history;
    if(transient) {
        frame_times.open(directory/"transient_times.txt");frame_times<<std::setprecision(17);
        frame_data.open(directory/"transient_temperature.bin",std::ios::binary);
        time_history.open(directory/"time_history.tsv");
        time_history<<std::setprecision(17)<<"time_s\tstep\tinner_iterations\tmaximum_temperature_K\tpower_W\tconvection_W\tradiation_W\tpcb_W\tstorage_W\tenergy_relative_error\n";
    }
    auto snapshot=[&](double t) {
        absolute_temperature<<<blocks(in.n),256>>>(in.n,theta.p,c.ambient,kelvin.p);
        auto values=kelvin.download();frame_times<<t<<'\n';
        frame_data.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));
        frame_data.flush();frame_times.flush();last_snapshot=t;
    };
    if(transient)snapshot(0);
    do {
        ++step;converged=false;
        double target=transient?std::min(time+value("time_step_s"),value("duration_s")):0;
        if(transient && value("duration_s")-target <= 1e-12*value("duration_s"))target=value("duration_s");
        active_dt=transient?target-time:0;
        if(transient)check(cudaMemcpy(old_time.p,theta.p,in.n*sizeof(double),cudaMemcpyDeviceToDevice));
        for(outer=1;outer<=int(value("max_outer_iterations"));++outer) {
            ++total_outer;auto previous=components;assemble();
            check(cudaMemcpy(candidate.p,theta.p,in.n*sizeof(double),cudaMemcpyDeviceToDevice));
            double linear_abs=value("pcg_absolute_tolerance_w");
            if(transient)linear_abs=std::min(linear_abs,.1*value("nonlinear_relative_tolerance")*std::max(components[8],1e-12)/std::sqrt(double(in.n)));
            linear=pcg.solve(row.p,col.p,a.p,rhs.p,candidate.p,int(value("max_pcg_iterations")),transient?0:value("pcg_relative_tolerance"),linear_abs);
            delta.zero();invalid.zero();
            relax<<<blocks(in.n),256>>>(in.n,candidate.p,value("relaxation"),c.ambient,value("maximum_temperature_k"),theta.p,delta.p,invalid.p);
            if(invalid.download()[0])throw std::runtime_error("Temperature limit/nonfinite state: no validated thermal solution");
            change=delta.download()[0];stats();update_losses();components=loss.download();loss_change=0;
            for(int j=0;j<8;++j)loss_change=std::max(loss_change,std::abs(components[j]-previous[j]));
            flow.zero();heat_flow<<<blocks(in.nf),256>>>(in.nf,face.p,xyz.p,h.p,eps.p,pcb.p,theta.p,c.ambient,c.trad,flow.p);
            flows=flow.download();stored=0;
            if(transient) {
                storage.zero();stored_heat_rate<<<blocks(in.n),256>>>(in.n,capacity.p,theta.p,old_time.p,active_dt,storage.p);
                stored=storage.download()[0];
            }
            energy=std::abs(components[8]-flows[0]-flows[1]-flows[2]-stored)/std::max(components[8],1e-30);
            assemble();spmv<<<blocks(in.n),256>>>(in.n,row.p,col.p,a.p,theta.p,pcg.ap.p);
            residual<<<blocks(in.n),256>>>(in.n,rhs.p,pcg.ap.p,pcg.r.p);
            // Scale residual by physical source/boundary power, not the potentially
            // large M*T_old/dt right-hand side, which could hide spatial errors.
            double rhs_scale=0;
            if(transient) {
                // energy tolerance and temperature/loss tolerances also gate acceptance.
                rhs_scale=std::max(components[8]/std::sqrt(double(in.n)),1e-30);
            } else rhs_scale=std::max(sqrt(pcg.dot(rhs.p,rhs.p)),1e-30);
            nonlinear=sqrt(pcg.dot(pcg.r.p,pcg.r.p))/rhs_scale;
            history<<target<<'\t'<<step<<'\t'<<outer<<'\t'<<linear.iterations<<'\t'<<linear.relative_residual<<'\t'<<nonlinear<<'\t'<<change<<'\t'<<loss_change<<'\t'<<components[8]<<'\t'<<flows[0]<<'\t'<<flows[1]<<'\t'<<flows[2]<<'\t'<<stored<<'\t'<<energy<<'\n';history.flush();
            if(!linear.converged){reason="linear_solver_did_not_converge";break;}
            if(change<=value("temperature_tolerance_k")&&loss_change<=value("loss_tolerance_w")&&energy<=value("energy_relative_tolerance")&&nonlinear<=value("nonlinear_relative_tolerance")) {
                converged=true;reason=transient?"time_steps_converged":"converged";break;
            }
        }
        if(!converged) {
            if(reason!="linear_solver_did_not_converge")reason="maximum_outer_iterations";
            if(transient) {
                check(cudaMemcpy(theta.p,old_time.p,in.n*sizeof(double),cudaMemcpyDeviceToDevice));
                stats();update_losses();components=loss.download();
                stored=accepted_stored;energy=accepted_energy;
                flow.zero();heat_flow<<<blocks(in.nf),256>>>(in.nf,face.p,xyz.p,h.p,eps.p,pcb.p,theta.p,c.ambient,c.trad,flow.p);flows=flow.download();
            }
            break;
        }
        if(transient) {
            time=target;accepted_stored=stored;accepted_energy=energy;auto max_by_region=maxt.download();double maximum=*std::max_element(max_by_region.begin(),max_by_region.end());
            time_history<<time<<'\t'<<step<<'\t'<<outer<<'\t'<<maximum<<'\t'<<components[8]<<'\t'<<flows[0]<<'\t'<<flows[1]<<'\t'<<flows[2]<<'\t'<<stored<<'\t'<<energy<<'\n';time_history.flush();
            if(step%int(value("snapshot_stride"))==0||time>=value("duration_s"))snapshot(time);
        }
        std::cout<<(transient?"time ":"steady solve ")<<(transient?time:0)<<" s: nonlinear iterations "<<outer<<", loss "<<components[8]<<" W, energy error "<<energy<<std::endl;
    } while(transient && time<value("duration_s"));
    if(transient && time>last_snapshot)snapshot(time);
    absolute_temperature<<<blocks(in.n), 256>>>(in.n, theta.p, c.ambient, kelvin.p);
    auto temperature = kelvin.download();
    std::ofstream binary(directory / "temperature.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char *>(temperature.data()), temperature.size() * sizeof(double));
    Device<double> integrated(in.nr);
    integrated.zero();
    integrated_sources<<<blocks(in.nt), 256>>>(in.nt, region.p, volume.p, rv.p, power.p, integrated.p);
    auto powers = power.download(), means = mean.download(), maxs = maxt.download(), mins = negmin.download(), generated = integrated.download();
    for (int r = 1; r < in.nr; ++r)
        if (std::abs(generated[r] - powers[r]) > 1e-10 * std::max(1., powers[r]))
            throw std::runtime_error("Integrated source audit failed");
    const char *labels[] = {"high_side_conduction", "high_side_switching", "low_side_conduction", "dead_time_diode", "winding_dc", "winding_ac", "core_unvalidated_placeholder", "auxiliary_ic"};
    std::ofstream json(directory / "solution.json");
    json << std::setprecision(17);
    json << "{\n\"mode\":\""<<(transient?"transient":"steady_state")<<"\",\n\"steady_state\":"<<(transient?"false":"true")
         <<",\n\"time_s\":"<<time<<",\n\"requested_duration_s\":"<<(transient?value("duration_s"):0)
         <<",\n\"accepted_time_steps\":"<<(transient?(converged?step:step-1):0)
         <<",\n\"storage_w\":"<<stored<<",\n\"time_steps_attempted\":"<<(transient?step:0)<<",\n";
    json << "\"converged\":" << (converged ? "true" : "false") << ",\n\"status\":\"" << reason << "\",\n\"outer_iterations\":" << total_outer << ",\n\"total_loss_w\":" << components[8] << ",\n\"output_power_w\":" << c.vout * c.iout << ",\n\"efficiency\":" << (c.vout * c.iout) / (c.vout * c.iout + components[8]) << ",\n\"convection_w\":" << flows[0] << ",\n\"radiation_w\":" << flows[1] << ",\n\"pcb_w\":" << flows[2] << ",\n\"energy_balance_relative_error\":" << energy << ",\n\"nonlinear_relative_residual\":" << nonlinear << ",\n\"losses_w\":{";
    for (int j = 0; j < 8; ++j)
        json << (j ? "," : "") << '\"' << labels[j] << "\":" << components[j];
    json << "},\n\"regions\":[";
    bool first = true;
    for (int r = 1; r < in.nr; ++r)
        if (volumes[r] > 0)
        {
            if (!first)
                json << ',';
            first = false;
            json << "{\"id\":" << r << ",\"power_w\":" << powers[r] << ",\"integrated_source_w\":" << generated[r] << ",\"volume_m3\":" << volumes[r] << ",\"q_w_m3\":" << powers[r] / volumes[r] << ",\"mean_k\":" << means[r] << ",\"maximum_k\":" << maxs[r] << ",\"minimum_k\":" << -mins[r] << '}';
        }
    json << "]\n}\n";
    check(cudaGetLastError());
    return converged ? 0 : 2;
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr << "Usage: thermal output/thermal | --self-test\nPrepare inputs with uv run python/thermal.py --prepare-only\n";
            return 1;
        }
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (!count)
            throw std::runtime_error("No CUDA device; no CPU solver is provided");
        check(cudaSetDevice(0));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 6)
            throw std::runtime_error("Double-precision atomic assembly requires compute capability >= 6.0");
        std::cout << "CUDA thermal GPU: " << prop.name << std::endl;
        if (std::string(argv[1]) == "--self-test")
        {
            self_test();
            return 0;
        }
        return run(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "CUDA thermal failed: " << e.what() << '\n';
        return 1;
    }
}
