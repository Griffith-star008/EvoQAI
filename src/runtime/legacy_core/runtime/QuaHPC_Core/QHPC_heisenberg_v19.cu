#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include "QHPC_quantum_ir_v19.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

// ============================================================================
// Heisenberg-picture / Pauli-tracker engine — v15.
// ----------------------------------------------------------------------------
// MATH CORRECTNESS (vs v13):
//
//   For a parametric gate R_G(θ) = exp(-i θ G / 2), conjugation acts on a
//   Pauli P that ANTICOMMUTES with G as:
//
//       R_G(θ)^† P R_G(θ)  =  cos(θ) P  +  i sin(θ) G P              (1)
//
//   The angle is the FULL θ, not θ/2. The factor on the commutator branch is
//   +i sin(θ), not -i sin(θ/2).
//
//   v13 used θ/2 and a stray minus sign — wrong by a factor of 2 in argument
//   and wrong by a sign on the second branch. v14 fixes both.
//
//   For P that COMMUTES with G, R^† P R = P unchanged.
//
//   GP for the six (G,P) anticommuting pairs:
//       G=Z:  Z*X = +iY    Z*Y = -iX
//       G=X:  X*Y = +iZ    X*Z = -iY
//       G=Y:  Y*X = -iZ    Y*Z = +iX
//
//   So eq. (1) becomes:   cos(θ) P  +  i sin(θ) · (factor) · P_new
//   where the factor is ±i (canceling with the leading i to give ±sin(θ)).
//
// ============================================================================

struct PauliString {
    uint32_t x=0, y=0, z=0;
    float    coeff_re=1.f, coeff_im=0.f;
};

__host__ __device__ inline void pauli_conj_h(PauliString& p, int q){
    uint32_t mq = 1u<<q;
    uint32_t xb = (p.x & mq) >> q;
    uint32_t zb = (p.z & mq) >> q;
    bool yset = (p.y & mq) != 0;
    p.x = (p.x & ~mq) | (zb<<q);
    p.z = (p.z & ~mq) | (xb<<q);
    if(yset){
        p.coeff_re = -p.coeff_re;
        p.coeff_im = -p.coeff_im;
    }
}
__host__ __device__ inline void pauli_conj_s(PauliString& p, int q){
    uint32_t mq = 1u<<q;
    bool x = (p.x & mq);
    bool y = (p.y & mq);
    if(x && !y){ p.x &= ~mq; p.y |= mq; }
    else if(!x && y){
        p.y &= ~mq; p.x |= mq;
        p.coeff_re = -p.coeff_re;
        p.coeff_im = -p.coeff_im;
    }
}
__host__ __device__ inline void pauli_conj_cnot(PauliString& p, int c, int t){
    uint32_t mc = 1u<<c, mt = 1u<<t;
    if(p.x & mc) p.x ^= mt;
    if(p.z & mt) p.z ^= mc;
}
__host__ __device__ inline void pauli_conj_cz(PauliString& p, int a, int b){
    uint32_t ma = 1u<<a, mb = 1u<<b;
    if(p.x & ma) p.z ^= mb;
    if(p.x & mb) p.z ^= ma;
}

// Hash a Pauli string for map-based deduplication after a step.
struct PauliKey {
    uint32_t x,y,z;
    bool operator==(const PauliKey& o)const{return x==o.x&&y==o.y&&z==o.z;}
};
struct PauliKeyHash {
    size_t operator()(const PauliKey& k) const {
        size_t h = (size_t)k.x*0x9E3779B97F4A7C15ULL;
        h ^= (size_t)k.y + 0x9E3779B97F4A7C15ULL + (h<<6) + (h>>2);
        h ^= (size_t)k.z + 0x9E3779B97F4A7C15ULL + (h<<6) + (h>>2);
        return h;
    }
};

class HeisenbergEngine {
public:
    HeisenbergEngine(const QuantumIR& ir, const Observable& obs,
                     int max_strings=1<<18, float prune_eps=1e-9f,
                     cudaStream_t stream=nullptr)
        : ir_(ir), obs_(obs), max_strings_(max_strings),
          prune_eps_(prune_eps), stream_(stream){
        live_.reserve(obs.terms.size());
        for(const auto& t : obs.terms){
            PauliString ps;
            ps.x=t.x_mask; ps.y=t.y_mask; ps.z=t.z_mask;
            ps.coeff_re=t.coeff; ps.coeff_im=0.f;
            live_.push_back(ps);
        }
    }

    bool evolve_backward(const float* h_params){
        std::vector<int> deriv(live_.size(), -1);
        for(int gi=(int)ir_.insts.size()-1; gi>=0; --gi){
            const IRInst& g = ir_.insts[gi];
            if(g.dead || g.kind==IRKind::NOP) continue;
            switch(g.kind){
                case IRKind::H:    for(auto& p : live_) pauli_conj_h(p, g.q0); break;
                case IRKind::S:    for(auto& p : live_) pauli_conj_s(p, g.q0); break;
                case IRKind::CNOT: for(auto& p : live_) pauli_conj_cnot(p, g.q0, g.q1); break;
                case IRKind::CZ:   for(auto& p : live_) pauli_conj_cz(p, g.q0, g.q1); break;
                case IRKind::RZ: case IRKind::RX: case IRKind::RY:
                    if(!_split_param(deriv, g, h_params)) return false;
                    _prune();
                    if((int)live_.size() > max_strings_){
                        fprintf(stderr,"[Heis] string-pop %zu > cap %d after prune.\n",
                            live_.size(), max_strings_);
                        return false;
                    }
                    break;
                case IRKind::T:
                case IRKind::CRZ:
                    fprintf(stderr,"[Heis] gate kind %d not supported.\n",(int)g.kind);
                    return false;
                default: break;
            }
        }
        deriv_.swap(deriv);
        return true;
    }

    float expectation_from_zero() const {
        float acc = 0.f;
        for(size_t i=0;i<live_.size();++i){
            int d = deriv_.empty() ? -1 : deriv_[i];
            if(d >= 0) continue;  // gradient branch, not expectation
            if(live_[i].x==0 && live_[i].y==0){
                acc += live_[i].coeff_re;   // <0|Z^z|0> = +1 always for the Z-only string on |0...0>
            }
        }
        return acc;
    }

    void gradient_from_zero(std::vector<float>& out) const {
        out.assign(ir_.n_params, 0.f);
        for(size_t i=0;i<live_.size();++i){
            int d = deriv_.empty() ? -1 : deriv_[i];
            if(d < 0) continue;
            if(live_[i].x==0 && live_[i].y==0){
                out[d] += live_[i].coeff_re;
            }
        }
    }

    size_t live_count() const { return live_.size(); }

private:
    const QuantumIR& ir_;
    const Observable& obs_;
    int max_strings_;
    float prune_eps_;
    cudaStream_t stream_;
    std::vector<PauliString> live_;
    std::vector<int>         deriv_;

    // CORRECT formula:  R_G(θ)^† P R_G(θ) = cos(θ)·P + i·sin(θ)·G·P
    // when {G,P}=0; otherwise unchanged.
    bool _split_param(std::vector<int>& deriv,
                      const IRInst& g, const float* h_params){
        float theta = (g.param_idx>=0) ? h_params[g.param_idx] : g.angle;
        float c = std::cos(theta);          // FULL angle, FIXED (v13 used θ/2)
        float s = std::sin(theta);
        uint32_t mq = 1u<<g.q0;

        std::vector<PauliString> nx; nx.reserve(live_.size()*2);
        std::vector<int> nxd;        nxd.reserve(live_.size()*2);

        for(size_t i=0;i<live_.size();++i){
            PauliString P = live_[i];
            int d = deriv[i];
            bool px = (P.x & mq), py = (P.y & mq), pz = (P.z & mq);
            bool anti = false;
            switch(g.kind){
                case IRKind::RZ: anti = (px||py); break;
                case IRKind::RX: anti = (py||pz); break;
                case IRKind::RY: anti = (px||pz); break;
                default: break;
            }
            if(!anti){
                // Commutes -> passes through unchanged (R^† P R = P).
                nx.push_back(P); nxd.push_back(d);
                continue;
            }
            // Branch A: cos(θ)·P  (P mask unchanged)
            PauliString A = P;
            A.coeff_re *= c; A.coeff_im *= c;
            if(std::abs(A.coeff_re) + std::abs(A.coeff_im) > prune_eps_){
                nx.push_back(A); nxd.push_back(d);
            }

            // Branch B: i·sin(θ)·G·P  (mask = G*P with proper Pauli algebra)
            PauliString B = P;
            // Determine the (sign, target-bit-transform) for G·P:
            //   G=Z, P has X on q  -> G P = i Y -> mask: x->0,y->1; factor +i
            //   G=Z, P has Y on q  -> G P = -i X-> mask: y->0,x->1; factor -i
            //   G=X, P has Y on q  -> G P = i Z -> mask: y->0,z->1; factor +i
            //   G=X, P has Z on q  -> G P = -i Y-> mask: z->0,y->1; factor -i
            //   G=Y, P has X on q  -> G P = -i Z-> mask: x->0,z->1; factor -i
            //   G=Y, P has Z on q  -> G P = +i X-> mask: z->0,x->1; factor +i
            // The leading +i in eq. (1) combined with the ±i above gives:
            //   total factor on Branch B = (i)*(±i)*sin(θ) = ∓sin(θ)
            float branch_re = 0.f, branch_im = 0.f;
            switch(g.kind){
                case IRKind::RZ:
                    if(px){ B.x &= ~mq; B.y |= mq; /* +i  -> total = -sin */ branch_re = -s; }
                    else  { B.y &= ~mq; B.x |= mq; /* -i  -> total = +sin */ branch_re = +s; }
                    break;
                case IRKind::RX:
                    if(py){ B.y &= ~mq; B.z |= mq; branch_re = -s; }
                    else  { B.z &= ~mq; B.y |= mq; branch_re = +s; }
                    break;
                case IRKind::RY:
                    if(px){ B.x &= ~mq; B.z |= mq; branch_re = +s; }
                    else  { B.z &= ~mq; B.x |= mq; branch_re = -s; }
                    break;
                default: break;
            }
            float nr = B.coeff_re*branch_re - B.coeff_im*branch_im;
            float ni = B.coeff_re*branch_im + B.coeff_im*branch_re;
            B.coeff_re = nr; B.coeff_im = ni;
            if(std::abs(B.coeff_re) + std::abs(B.coeff_im) > prune_eps_){
                nx.push_back(B); nxd.push_back(d);
            }
        }
        live_.swap(nx); deriv.swap(nxd);
        return (int)live_.size() <= max_strings_;
    }

    // Pauli-term fusion: combine duplicate (x,y,z) masks by summing coefficients,
    // then drop terms whose magnitude falls below prune_eps_.
    void _prune(){
        std::vector<int> idx(live_.size()); std::iota(idx.begin(),idx.end(),0);
        std::sort(idx.begin(), idx.end(), [&](int a,int b){
            const auto& A=live_[a]; const auto& B=live_[b];
            if(A.x!=B.x) return A.x<B.x;
            if(A.y!=B.y) return A.y<B.y;
            return A.z<B.z;
        });
        std::vector<PauliString> out; out.reserve(live_.size());
        for(size_t k=0;k<idx.size();){
            PauliString acc = live_[idx[k]];
            size_t kk = k+1;
            while(kk<idx.size()){
                const auto& A=live_[idx[k]]; const auto& B=live_[idx[kk]];
                if(A.x==B.x && A.y==B.y && A.z==B.z){
                    acc.coeff_re += B.coeff_re;
                    acc.coeff_im += B.coeff_im;
                    ++kk;
                } else break;
            }
            if(std::abs(acc.coeff_re)+std::abs(acc.coeff_im) > prune_eps_){
                out.push_back(acc);
            }
            k = kk;
        }
        live_.swap(out);
    }
};
