// Linear magnetostatic H(curl) edge FEM. No conductivity/eddy-current operator.
// Exact algebraic gauge: add alpha G G^T, with nodal potentials zero on boundary.
// Closed-current RHS satisfies G^T f=0, so this fixes A without changing curl(A).
#include <cuda_runtime.h>
#include "module_geometry.hpp"
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
constexpr double MU0 = 1.2566370614359172954e-6;
void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T>
struct Device
{
    T *p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count)
    {
        if (n)
            check(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Device()
    {
        if (p)
            cudaFree(p);
    }
    Device(const Device &) = delete;
    void zero()
    {
        if (n)
            check(cudaMemset(p, 0, n * sizeof(T)));
    }
    void put(const std::vector<T> &v)
    {
        if (v.size() != n)
            throw std::runtime_error("Device size mismatch");
        if (n)
            check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const
    {
        std::vector<T> v(n);
        if (n)
            check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
template <class T>
std::vector<T> read(std::ifstream &f, size_t n)
{
    std::vector<T> v(n);
    if (n && !f.read((char *)v.data(), n * sizeof(T)))
        throw std::runtime_error("Truncated EM input");
    return v;
}
template <class T>
void write(const fs::path &p, const std::vector<T> &v)
{
    std::ofstream f(p, std::ios::binary);
    if (!f.write((const char *)v.data(), v.size() * sizeof(T)))
        throw std::runtime_error("Cannot write EM output");
}
struct Input
{
    int n = 0, nt = 0, ne = 0, ns = 0, nf = 0, nframes = 0;
    std::vector<double> xyz, mur, sigma, sa, sb, weight, times;
    std::vector<int> tet, edges, te, fixed, source, faces, owners;
    Input() = default;
    explicit Input(const fs::path &p)
    {
        std::ifstream f(p, std::ios::binary);
        char magic[8];
        if (!f.read(magic, 8) || std::string(magic, 8) != "POLEMG01")
            throw std::runtime_error("Invalid EM input");
        auto c = read<int>(f, 6);
        n = c[0];
        nt = c[1];
        ne = c[2];
        ns = c[3];
        nf = c[4];
        nframes = c[5];
        if (std::min({n, nt, ne, nf, nframes}) <= 0 || ns < 0 || std::max({n, nt, ne, ns, nf, nframes}) > 100000000)
            throw std::runtime_error("Invalid EM counts");
        xyz = read<double>(f, size_t(n) * 3);
        tet = read<int>(f, size_t(nt) * 4);
        edges = read<int>(f, size_t(ne) * 2);
        te = read<int>(f, size_t(nt) * 6);
        fixed = read<int>(f, ne);
        mur = read<double>(f, nt);
        sigma = read<double>(f, nt);
        source = read<int>(f, ns);
        sa = read<double>(f, size_t(ns) * 4);
        sb = read<double>(f, size_t(ns) * 4);
        weight = read<double>(f, ns);
        times = read<double>(f, nframes);
        faces = read<int>(f, size_t(nf) * 3);
        owners = read<int>(f, size_t(nf) * 2);
        for (int i : tet)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid tetrahedron index");
        for (int i : edges)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid edge node");
        for (int i : te)
            if (i == 0 || std::abs(i) > ne)
                throw std::runtime_error("Invalid oriented edge");
        for (int i : source)
            if (i < 0 || i >= nt)
                throw std::runtime_error("Invalid source tetrahedron");
        for (int i : faces)
            if (i < 0 || i >= n)
                throw std::runtime_error("Invalid face node");
        for (int i : owners)
            if (i < -1 || i >= nt)
                throw std::runtime_error("Invalid face owner");
        for (int i : fixed)
            if (i != 0 && i != 1)
                throw std::runtime_error("Invalid constraint mask");
        for (double m : mur)
            if (!(m > 0) || !std::isfinite(m))
                throw std::runtime_error("Invalid permeability");
        for (double v : xyz)
            if (!std::isfinite(v))
                throw std::runtime_error("Nonfinite mesh coordinate");
        for (double v : weight)
            if (!(v > 0) || !std::isfinite(v))
                throw std::runtime_error("Invalid source weight");
        for (const auto *values : {&sa, &sb})
            for (double v : *values)
                if (!std::isfinite(v) || v < -1e-8 || v > 1 + 1e-8)
                    throw std::runtime_error("Invalid source barycentric coordinate");
        for (int s = 0; s < ns; ++s)
            for (const auto *values : {&sa, &sb})
            {
                double sum = 0;
                for (int j = 0; j < 4; ++j)
                    sum += (*values)[4 * s + j];
                if (std::abs(sum - 1) > 1e-8)
                    throw std::runtime_error("Invalid barycentric sum");
            }
        if (times.front() != 0 || !std::is_sorted(times.begin(), times.end()))
            throw std::runtime_error("Invalid frame times");
        for (double t : times)
            if (!std::isfinite(t) || t < 0)
                throw std::runtime_error("Invalid frame time");
    }
};
__host__ __device__ double3 sub(double3 a, double3 b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ double3 mul(double3 a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ double3 add(double3 a, double3 b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ double dot3(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ double3 cross3(double3 a, double3 b) { return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ double3 vec(const double *p, int i) { return make_double3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
__device__ void store(double *p, int i, double3 v)
{
    p[3 * i] = v.x;
    p[3 * i + 1] = v.y;
    p[3 * i + 2] = v.z;
}
__global__ void geometry(int nt, const double *xyz, const int *tet, const int *te, const double *mur, double *curls, double *vol, double *coef, double *diag, int *bad)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt)
        return;
    double3 p = vec(xyz, tet[4 * t]), a = sub(vec(xyz, tet[4 * t + 1]), p), b = sub(vec(xyz, tet[4 * t + 2]), p), c = sub(vec(xyz, tet[4 * t + 3]), p);
    double det = dot3(a, cross3(b, c));
    if (!(det > 0) || !isfinite(det))
    {
        atomicAdd(bad, 1);
        return;
    }
    double3 g[4];
    g[1] = mul(cross3(b, c), 1 / det);
    g[2] = mul(cross3(c, a), 1 / det);
    g[3] = mul(cross3(a, b), 1 / det);
    g[0] = mul(add(add(g[1], g[2]), g[3]), -1);
    vol[t] = det / 6;
    coef[t] = vol[t] / (MU0 * mur[t]);
    int k = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j, ++k)
        {
            int e = te[6 * t + k];
            double3 curl = mul(cross3(g[i], g[j]), e > 0 ? 2 : -2);
            store(curls, 6 * t + k, curl);
            atomicAdd(diag + abs(e) - 1, coef[t] * dot3(curl, curl));
        }
}
__global__ void assemble_source(int ns, const int *source, const int *te, const double *a, const double *b, const double *w, double current, double *rhs)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ns)
        return;
    int k = 0, t = source[s];
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j, ++k)
        {
            int e = te[6 * t + k];
            double integral = (a[4 * s + i] * b[4 * s + j] - a[4 * s + j] * b[4 * s + i]) * w[s] * current;
            atomicAdd(rhs + abs(e) - 1, e > 0 ? integral : -integral);
        }
}
__global__ void divergence_current(int ne, const int *edges, const double *rhs, double *div)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < ne)
    {
        atomicAdd(div + edges[2 * e], -rhs[e]);
        atomicAdd(div + edges[2 * e + 1], rhs[e]);
    }
}
__global__ void apply_elements(int nt, const int *te, const double *curls, const double *coef, const double *x, double *y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt)
        return;
    double3 B = make_double3(0, 0, 0);
    for (int k = 0; k < 6; ++k)
        B = add(B, mul(vec(curls, 6 * t + k), x[abs(te[6 * t + k]) - 1]));
    for (int k = 0; k < 6; ++k)
        atomicAdd(y + abs(te[6 * t + k]) - 1, coef[t] * dot3(vec(curls, 6 * t + k), B));
}
__global__ void constrain(int n, const int *fixed, const double *x, double *y)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n && fixed[e])
        y[e] = x ? x[e] : 0;
}
__global__ void gauge_div(int ne, const int *edges, const int *boundary, const double *x, double *d)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne)
        return;
    int a = edges[2 * e], b = edges[2 * e + 1];
    if (!boundary[a])
        atomicAdd(d + a, -x[e]);
    if (!boundary[b])
        atomicAdd(d + b, x[e]);
}
__global__ void gauge_add(int ne, const int *edges, const int *boundary, const double *d, double alpha, double *y)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne)
        return;
    int a = edges[2 * e], b = edges[2 * e + 1];
    y[e] += alpha * ((boundary[b] ? 0 : d[b]) - (boundary[a] ? 0 : d[a]));
}
__global__ void gauge_diag(int ne, const int *edges, const int *boundary, double alpha, double *diag)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < ne)
        diag[e] += alpha * ((boundary[edges[2 * e]] ? 0 : 1) + (boundary[edges[2 * e + 1]] ? 0 : 1));
}
__global__ void precondition(int n, const double *r, const double *diag, const int *fixed, double *z)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n)
        z[e] = fixed[e] ? r[e] : r[e] / diag[e];
}
__global__ void axpy(int n, double a, const double *x, double *y)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n)
        y[e] += a * x[e];
}
__global__ void direction(int n, const double *z, double beta, double *p)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n)
        p[e] = z[e] + beta * p[e];
}
__global__ void reduce_dot(int n, const double *a, const double *b, double *out)
{
    __shared__ double v[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    for (; i < n; i += blockDim.x * gridDim.x)
        sum += a[i] * b[i];
    v[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2)
    {
        if (threadIdx.x < s)
            v[threadIdx.x] += v[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, v[0]);
}
__global__ void recover(int nt, const int *te, const double *curls, const double *x, double *B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt)
        return;
    double3 b = make_double3(0, 0, 0);
    for (int k = 0; k < 6; ++k)
        b = add(b, mul(vec(curls, 6 * t + k), x[abs(te[6 * t + k]) - 1]));
    store(B, t, b);
}
__global__ void face_flux(int nf, const int *faces, const int *owners, const double *xyz, const double *B, double *jump, double *area)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf)
        return;
    double3 a = vec(xyz, faces[3 * f]), b = vec(xyz, faces[3 * f + 1]), c = vec(xyz, faces[3 * f + 2]);
    double3 normal = mul(cross3(sub(b, a), sub(c, a)), .5);
    int l = owners[2 * f], r = owners[2 * f + 1];
    double3 d = vec(B, l);
    if (r >= 0)
        d = sub(d, vec(B, r));
    jump[f] = dot3(normal, d);
    area[f] = sqrt(dot3(normal, normal));
}
__global__ void waveform(int nf, const double *times, double vin, double vout, double inductance, double frequency, double mean, double *current)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf)
        return;
    double d = vout / vin, ripple = vout * (1 - d) / (inductance * frequency), phase = times[i] * frequency;
    phase -= floor(phase);
    current[i] = mean - .5 * ripple + ripple * (phase < d ? phase / d : (1 - phase) / (1 - d));
}
__global__ void scale_field(int n, const double *base, const double *current, int frame, double mean, double *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = base[i] * current[frame] / mean;
}
int blocks(int n) { return (n + 255) / 256; }
struct Solver
{
    const Input &h;
    double gauge_alpha = 0;
    Device<double> xyz, mur, curls, vol, coef, diag, rhs, x, r, z, p, q, tmp, nodal;
    Device<int> tet, te, fixed, bad, edges, boundary;
    Solver(const Input &in) : h(in), xyz(in.xyz.size()), mur(in.mur.size()), curls(size_t(in.nt) * 18), vol(in.nt), coef(in.nt), diag(in.ne), rhs(in.ne), x(in.ne), r(in.ne), z(in.ne), p(in.ne), q(in.ne), tmp(1), nodal(in.n), tet(in.tet.size()), te(in.te.size()), fixed(in.fixed.size()), bad(1), edges(in.edges.size()), boundary(in.n)
    {
        std::vector<int> bn(in.n, 0);
        for (int e = 0; e < in.ne; ++e)
            if (in.fixed[e])
            {
                bn[in.edges[2 * e]] = 1;
                bn[in.edges[2 * e + 1]] = 1;
            }
        boundary.put(bn);
        edges.put(in.edges);
        xyz.put(in.xyz);
        mur.put(in.mur);
        tet.put(in.tet);
        te.put(in.te);
        fixed.put(in.fixed);
        diag.zero();
        bad.zero();
        geometry<<<blocks(h.nt), 256>>>(h.nt, xyz.p, tet.p, te.p, mur.p, curls.p, vol.p, coef.p, diag.p, bad.p);
        check(cudaGetLastError());
        if (bad.get()[0])
            throw std::runtime_error("Nonpositive/invalid tetrahedron volume");
        auto d = diag.get();
        for (int i = 0; i < h.ne; ++i)
            if (!h.fixed[i] && (!(d[i] > 0) || !std::isfinite(d[i])))
                throw std::runtime_error("Invalid operator diagonal");
        double sum = 0;
        for (double v : d)
            sum += v;
        gauge_alpha = sum / h.ne;
        gauge_diag<<<blocks(h.ne), 256>>>(h.ne, edges.p, boundary.p, gauge_alpha, diag.p);
    }
    double dot(const Device<double> &a, const Device<double> &b)
    {
        tmp.zero();
        reduce_dot<<<std::min(blocks(h.ne), 1024), 256>>>(h.ne, a.p, b.p, tmp.p);
        return tmp.get()[0];
    }
    void apply(const Device<double> &a, Device<double> &b)
    {
        b.zero();
        apply_elements<<<blocks(h.nt), 256>>>(h.nt, te.p, curls.p, coef.p, a.p, b.p);
        nodal.zero();
        gauge_div<<<blocks(h.ne), 256>>>(h.ne, edges.p, boundary.p, a.p, nodal.p);
        gauge_add<<<blocks(h.ne), 256>>>(h.ne, edges.p, boundary.p, nodal.p, gauge_alpha, b.p);
        constrain<<<blocks(h.ne), 256>>>(h.ne, fixed.p, a.p, b.p);
    }
    struct Result
    {
        int iterations;
        double relative_residual, absolute_residual;
    };
    Result solve(int maxit, double rtol, double atol)
    {
        constrain<<<blocks(h.ne), 256>>>(h.ne, fixed.p, nullptr, rhs.p);
        x.zero();
        check(cudaMemcpy(r.p, rhs.p, h.ne * sizeof(double), cudaMemcpyDeviceToDevice));
        double norm = std::sqrt(dot(rhs, rhs)), target = std::max(atol, rtol * norm);
        if (!(norm > 0) || !std::isfinite(norm))
            throw std::runtime_error("Zero/nonfinite excitation");
        precondition<<<blocks(h.ne), 256>>>(h.ne, r.p, diag.p, fixed.p, z.p);
        check(cudaMemcpy(p.p, z.p, h.ne * sizeof(double), cudaMemcpyDeviceToDevice));
        double rz = dot(r, z);
        int iterations = 0;
        for (; iterations < maxit; ++iterations)
        {
            apply(p, q);
            double pq = dot(p, q);
            if (!(pq > 0) || !std::isfinite(pq))
                throw std::runtime_error("PCG breakdown: check gauge/topology");
            double alpha = rz / pq;
            axpy<<<blocks(h.ne), 256>>>(h.ne, alpha, p.p, x.p);
            axpy<<<blocks(h.ne), 256>>>(h.ne, -alpha, q.p, r.p);
            double residual = std::sqrt(dot(r, r));
            if (!std::isfinite(residual))
                throw std::runtime_error("Nonfinite PCG residual");
            bool refresh = residual <= target;
            if (refresh)
            { // Recompute true residual; restart direction if recursive residual drifted.
                apply(x, q);
                check(cudaMemcpy(r.p, rhs.p, h.ne * sizeof(double), cudaMemcpyDeviceToDevice));
                axpy<<<blocks(h.ne), 256>>>(h.ne, -1, q.p, r.p);
                residual = std::sqrt(dot(r, r));
            }
            if (residual <= target)
                return {iterations + 1, residual / norm, residual};
            if (iterations % 1000 == 999)
                std::cout << "PCG " << iterations + 1 << " residual " << residual / norm << std::endl;
            precondition<<<blocks(h.ne), 256>>>(h.ne, r.p, diag.p, fixed.p, z.p);
            double next = dot(r, z);
            if (!(next > 0) || !std::isfinite(next))
                throw std::runtime_error("PCG preconditioner breakdown");
            direction<<<blocks(h.ne), 256>>>(h.ne, z.p, refresh ? 0 : next / rz, p.p);
            rz = next;
        }
        throw std::runtime_error("Magnetostatic PCG did not converge; no field result written");
    }
};
std::map<std::string, double> params(const fs::path &p)
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
        throw std::runtime_error("Malformed parameters");
    return m;
}
void selftest()
{
    Input h;
    h.n = 4;
    h.nt = 1;
    h.ne = 6;
    h.xyz = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    h.tet = {0, 1, 2, 3};
    h.edges = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
    h.te = {1, 2, 3, 4, 5, 6};
    h.fixed = {1, 1, 1, 0, 0, 0};
    h.mur = {1};
    Solver s(h);
    Device<double> known(6);
    known.put({0, 0, 0, 2e-6, -1e-6, 3e-6});
    s.apply(known, s.rhs);
    s.solve(100, 1e-12, 1e-15);
    auto x = s.x.get();
    auto k = known.get();
    double err = 0;
    for (int i = 0; i < 6; ++i)
        err = std::max(err, std::abs(x[i] - k[i]));
    if (err > 1e-15)
        throw std::runtime_error("Edge operator/PCG manufactured test failed");
    auto rhs = s.rhs.get();
    for (auto &v : rhs)
        v *= 2;
    s.rhs.put(rhs);
    s.solve(100, 1e-12, 1e-15);
    auto twice = s.x.get();
    for (int i = 0; i < 6; ++i)
        if (std::abs(twice[i] - 2 * x[i]) > 1e-15)
            throw std::runtime_error("Linearity self-test failed");
    // Exact sequence: edge integrals of a nodal gradient must have zero curl.
    h.fixed.assign(6, 0);
    Solver g(h);
    std::vector<double> phi = {1, 3, -2, 4}, grad(6);
    for (int i = 0; i < 6; ++i)
        grad[i] = phi[h.edges[2 * i + 1]] - phi[h.edges[2 * i]];
    g.x.put(grad);
    Device<double> B(3);
    recover<<<1, 256>>>(1, g.te.p, g.curls.p, g.x.p, B.p);
    for (double b : B.get())
        if (std::abs(b) > 1e-12)
            throw std::runtime_error("curl grad != 0");
    auto curls = g.curls.get();
    if (std::abs(curls[9 + 2] - 2) > 1e-12)
        throw std::runtime_error("Edge curl orientation test failed");
    Device<double> times(4), current(4);
    times.put({0, 1.0 / 12e6, 1e-6, 2e-6});
    waveform<<<1, 256>>>(4, times.p, 12, 1, 220e-9, 1e6, 20, current.p);
    auto wave = current.get();
    if (std::abs(wave[0] - (20 - 25. / 12)) > 1e-10 || std::abs(wave[1] - (20 + 25. / 12)) > 1e-10 || std::abs(wave[2] - wave[0]) > 1e-10)
        throw std::runtime_error("Buck waveform test failed");
    check(cudaDeviceSynchronize());
    std::cout << "PASS: edge curl, curl(grad)=0, gauged PCG, linearity, and duty-cycle waveform\n";
}
int main(int argc, char **argv)
{
    fs::path out;
    try
    {
        if (argc == 3 && std::string(argv[1]) == "--model")
        {
            auto model = load_module_geometry(argv[2]);
            std::cout << "Geometry inspection only: " << model.boxes.size() << " boxes, " << model.capsules.size() << " winding capsules. Use python/em.py to prepare the air-domain edge FEM input.\n";
            return 0;
        }
        if (argc != 2)
        {
            std::cerr << "Usage: em --self-test | em OUTPUT_DIRECTORY\n";
            return 1;
        }
        bool test = std::string(argv[1]) == "--self-test";
        if (!test)
        {
            out = argv[1];
            for (const char *f : {"B_dc.bin", "A_edges.bin", "B_series.bin", "current.bin", "times.bin", "solution.json"})
                fs::remove(out / f);
        }
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (count < 1)
            throw std::runtime_error("No CUDA device available; no CPU solver fallback");
        check(cudaSetDevice(0));
        check(cudaFree(nullptr));
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        if (device.major < 6)
            throw std::runtime_error("Double atomicAdd requires compute capability >=6.0");
        if (test)
        {
            selftest();
            return 0;
        }
        Input h(out / "input.bin");
        auto m = params(out / "parameters.txt");
        double mean = m.at("current_a");
        if (!(mean > 0) || !(m.at("vin_v") > m.at("vout_v")) || !(m.at("vout_v") > 0) || !(m.at("inductance_h") > 0) || !(m.at("frequency_hz") > 0))
            throw std::runtime_error("Invalid buck operating point");
        if (h.ns == 0 && !fs::exists(out / "rhs_edges.bin"))
            throw std::runtime_error("Distributed source requires rhs_edges.bin");
        Solver s(h);
        Device<int> source(h.ns), edges(h.edges.size());
        Device<double> a(h.sa.size()), b(h.sb.size()), weights(h.ns), div(h.n);
        source.put(h.source);
        a.put(h.sa);
        b.put(h.sb);
        weights.put(h.weight);
        edges.put(h.edges);
        s.rhs.zero();
        if (fs::exists(out / "rhs_edges.bin"))
        {
            std::ifstream input_rhs(out / "rhs_edges.bin", std::ios::binary);
            auto values = read<double>(input_rhs, h.ne);
            for (double &v : values)
            {
                if (!std::isfinite(v)) throw std::runtime_error("Nonfinite distributed source");
                v *= mean;
            }
            s.rhs.put(values);
        }
        else
        {
            assemble_source<<<blocks(h.ns), 256>>>(h.ns, source.p, s.te.p, a.p, b.p, weights.p, mean, s.rhs.p);
        }
        div.zero();
        divergence_current<<<blocks(h.ne), 256>>>(h.ne, edges.p, s.rhs.p, div.p);
        double continuity = 0;
        for (double v : div.get())
            continuity = std::max(continuity, std::abs(v) / mean);
        if (continuity > 1e-8)
            throw std::runtime_error("Source violates discrete current continuity");
        auto result = s.solve(int(m.at("max_iterations")), m.at("relative_tolerance"), m.at("absolute_tolerance"));
        Device<double> B(size_t(h.nt) * 3);
        recover<<<blocks(h.nt), 256>>>(h.nt, s.te.p, s.curls.p, s.x.p, B.p);
        check(cudaGetLastError());
        auto dc = B.get();
        double maximum = 0;
        for (int t = 0; t < h.nt; ++t)
        {
            double v = std::sqrt(dc[3 * t] * dc[3 * t] + dc[3 * t + 1] * dc[3 * t + 1] + dc[3 * t + 2] * dc[3 * t + 2]);
            if (!std::isfinite(v))
                throw std::runtime_error("Nonfinite field");
            maximum = std::max(maximum, v);
        }
        Device<int> faces(h.faces.size()), owners(h.owners.size());
        faces.put(h.faces);
        owners.put(h.owners);
        Device<double> jumps(h.nf), areas(h.nf);
        face_flux<<<blocks(h.nf), 256>>>(h.nf, faces.p, owners.p, s.xyz.p, B.p, jumps.p, areas.p);
        auto jump = jumps.get(), area = areas.get();
        double flux = 0;
        for (int i = 0; i < h.nf; ++i)
            flux = std::max(flux, std::abs(jump[i]) / (area[i] * std::max(maximum, 1e-300)));
        // Second independently solved current checks the linearity of computed B, not just displayed colors.
        auto first = s.x.get(), rhs = s.rhs.get();
        for (double &v : rhs)
            v *= 2;
        s.rhs.put(rhs);
        s.solve(int(m.at("max_iterations")), m.at("relative_tolerance"), m.at("absolute_tolerance"));
        Device<double> B2(size_t(h.nt) * 3);
        recover<<<blocks(h.nt), 256>>>(h.nt, s.te.p, s.curls.p, s.x.p, B2.p);
        auto twice = B2.get();
        double scale_error = 0;
        for (size_t i = 0; i < dc.size(); ++i)
            scale_error = std::max(scale_error, std::abs(twice[i] - 2 * dc[i]) / (2 * std::max(maximum, 1e-300)));
        if (flux > 1e-8 || scale_error > 1e-7)
            throw std::runtime_error("Flux continuity or independent-current linearity audit failed");
        Device<double> times(h.nframes), current(h.nframes), frame(dc.size());
        times.put(h.times);
        waveform<<<blocks(h.nframes), 256>>>(h.nframes, times.p, m.at("vin_v"), m.at("vout_v"), m.at("inductance_h"), m.at("frequency_hz"), mean, current.p);
        // Write to temporary file; failed runs never publish a partial field series.
        fs::path temporary = out / "B_series.partial";
        std::ofstream series(temporary, std::ios::binary);
        for (int i = 0; i < h.nframes; ++i)
        {
            scale_field<<<blocks(int(dc.size())), 256>>>(int(dc.size()), B.p, current.p, i, mean, frame.p);
            auto v = frame.get();
            if (!series.write((const char *)v.data(), v.size() * sizeof(double)))
                throw std::runtime_error("Cannot write field series");
        }
        series.close();
        write(out / "B_dc.bin", dc);
        write(out / "A_edges.bin", first);
        write(out / "times.bin", h.times);
        write(out / "current.bin", current.get());
        fs::rename(temporary, out / "B_series.bin");
        std::ofstream report(out / "solution.json");
        report << std::setprecision(17) << "{\n\"converged\":true,\n\"mode\":\"quasistatic_no_eddy\",\n\"gpu\":\"" << device.name << "\",\n\"iterations\":" << result.iterations << ",\n\"relative_residual\":" << result.relative_residual << ",\n\"source_continuity_relative\":" << continuity << ",\n\"normal_flux_jump_relative\":" << flux << ",\n\"independent_current_linearity_relative\":" << scale_error << ",\n\"maximum_B_T\":" << maximum << ",\n\"eddy_currents_included\":false\n}\n";
        std::cout << "Converged magnetostatic edge FEM: " << result.iterations << " iterations; relative residual " << result.relative_residual << "; max B " << maximum << " T. Time series: linear quasistatic approximation, NO eddy currents.\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        if (!out.empty())
        {
            for (const char *f : {"B_dc.bin", "A_edges.bin", "B_series.bin", "B_series.partial", "current.bin", "times.bin", "solution.json"})
            {
                std::error_code ec;
                fs::remove(out / f, ec);
            }
        }
        std::cerr << "EM error: " << e.what() << "\n";
        return 1;
    }
}
