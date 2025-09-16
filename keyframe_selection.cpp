// adaptive_keyframe_selection.cpp
// g++ -std=c++17 adaptive_keyframe_selection.cpp -O2 -I /usr/include/eigen3 -o avks

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// -------------------- SO(3) 
static inline Mat3 hat(const Vec3& w) {
    Mat3 W;
    W <<     0, -w.z(),  w.y(),
          w.z(),     0, -w.x(),
         -w.y(),  w.x(),     0;
    return W;
}

// R = Exp(omega)
static inline Mat3 ExpSO3(const Vec3& omega) {
    double theta = omega.norm();
    if (theta < 1e-12) {
        return Mat3::Identity();
    }
    Vec3 a = omega / theta;
    Mat3 Ahat = hat(a);
    return Mat3::Identity()
         + std::sin(theta) * Ahat
         + (1.0 - std::cos(theta)) * (Ahat * Ahat);
}

// -------------------- IMU --------------------
struct ImuBias {
    Vec3 ba{Vec3::Zero()}; //  b_t^a
    Vec3 bg{Vec3::Zero()}; //  b_t^ω
};

struct State {
    Vec3 p{Vec3::Zero()};  //  x_t
    Vec3 v{Vec3::Zero()};  //  v_t
    Mat3 R{Mat3::Identity()}; //  R_t
};

// 
// v_{t+dt} = v_t + g*dt + R_t (a_t - b^a_t - n^a_t) dt
// x_{t+dt} = x_t + v_t*dt + 1/2 g dt^2 + 1/2 R_t (a_t - b^a_t - n^a_t) dt^2
// R_{t+dt} = R_t * Exp((w_t - b^ω_t - n^ω_t) dt)
State propagateImu(
    const State& st,
    const Vec3& a_meas,      // IMU  a_t
    const Vec3& w_meas,      // IMU  ω_t
    const ImuBias& bias,
    double dt,
    const Vec3& g = Vec3(0,0,-9.81),
    const Vec3& na = Vec3::Zero(),   // 
    const Vec3& nw = Vec3::Zero()    // 
) {
    Vec3 a_hat = a_meas - bias.ba - na;
    Vec3 w_hat = w_meas - bias.bg - nw;

    State ns = st;
    ns.v = st.v + g*dt + st.R * a_hat * dt;
    ns.p = st.p + st.v*dt + 0.5*g*dt*dt + 0.5*st.R*a_hat*dt*dt;
    ns.R = st.R * ExpSO3(w_hat * dt);
    return ns;
}

// --------------------  Chamfer  --------------------
// d(P1,P2) = (1/|P1|) Σ_{a∈P1} min_{b∈P2} ||a-b||^2
//          + (1/|P2|) Σ_{b∈P2} min_{a∈P1} ||a-b||^2
double chamferSymmetric(
    const std::vector<Vec3>& P1,
    const std::vector<Vec3>& P2
) {
    auto oneWay = [](const std::vector<Vec3>& A, const std::vector<Vec3>& B){
        if (A.empty() || B.empty()) return std::numeric_limits<double>::infinity();
        double acc = 0.0;
        for (const auto& a : A) {
            double m = std::numeric_limits<double>::infinity();
            for (const auto& b : B) {
                double d2 = (a - b).squaredNorm();
                if (d2 < m) m = d2;
            }
            acc += m;
        }
        return acc / static_cast<double>(A.size());
    };
    return oneWay(P1, P2) + oneWay(P2, P1);
}

// -------------------- keyframe score --------------------
// 文中： Q = φ * ( M_c * O_r / M_r + D_c * O_r / D_r )
// 其中 φ = s + γ - β - α
// γ = (2 - N/2) * (1 - M_c/M_r)
// β =  O_c/M_c - O_c/M_r
// α =  O_c/M_c - 0.5
// 变量含义：
//   M_c, M_r: 
//   D_c, D_r: 
//   O_r:      
//   O_c:      
//   N:        
//   s:        
struct KeyframeScoreInputs {
    int Mc{}, Mr{};
    int Dc{}, Dr{};
    int Or{};     // reference 
    int Oc{};     // current   
    int N{};      // 
    double s{1.0}; // 
};

struct ScoreDetail {
    double phi{};
    double gamma{};
    double beta{};
    double alpha{};
    double term_feat{};
    double Q{};
};

ScoreDetail computeKeyframeScore(const KeyframeScoreInputs& in) {
    ScoreDetail sd;

    // 
    auto safeDiv = [](double a, double b){
        return (std::abs(b) < 1e-12) ? 0.0 : (a / b);
    };

    sd.gamma = (2.0 - 0.5*static_cast<double>(in.N))
               * (1.0 - safeDiv(in.Mc, in.Mr));

    sd.beta  = safeDiv(in.Oc, in.Mc) - safeDiv(in.Oc, in.Mr);
    sd.alpha = safeDiv(in.Oc, in.Mc) - 0.5;

    sd.phi = in.s + sd.gamma - sd.beta - sd.alpha;

    // 
    double t1 = safeDiv(static_cast<double>(in.Mc*in.Or), in.Mr);
    double t2 = safeDiv(static_cast<double>(in.Dc*in.Or), in.Dr);
    sd.term_feat = t1 + t2;

    sd.Q = sd.phi * sd.term_feat;
    return sd;
}

bool shouldCreateKeyframe(
    const ScoreDetail& sd,
    double threshold_Q   //   s * (0.5 * Mr) 
){
    return sd.Q >= threshold_Q;
}

// --------------------  --------------------
int main() {
    // 1) 
    State st0;
    st0.p = Vec3(0,0,0);
    st0.v = Vec3(0,0,0);
    st0.R = Mat3::Identity();

    ImuBias bias;
    Vec3 a_meas(0.0, 0.0, 0.0);  // 
    Vec3 w_meas(0.0, 0.0, 0.0);
    double dt = 0.01;

    State st1 = propagateImu(st0, a_meas, w_meas, bias, dt);
    std::cout << "[IMU] p=" << st1.p.transpose()
              << " v=" << st1.v.transpose() << "\n";

    // 2) Chamfer 
    std::vector<Vec3> P1{{0,0,0},{1,0,0},{0,1,0}};
    std::vector<Vec3> P2{{0,0,0},{1,1,0}};
    double cd = chamferSymmetric(P1, P2);
    std::cout << "[Chamfer] d=" << cd << "\n";

    
    KeyframeScoreInputs in;
    in.Mc=120; in.Mr=150;
    in.Dc=400; in.Dr=420;
    in.Or=30;  in.Oc=25;
    in.N=8;    in.s=1.0;

    ScoreDetail sd = computeKeyframeScore(in);
    std::cout << "[Score] phi=" << sd.phi
              << " term=" << sd.term_feat
              << " Q=" << sd.Q << "\n";

    double threshold_Q = 50.0; // 示例阈值（项目中请基于数据标定）
    bool makeKF = shouldCreateKeyframe(sd, threshold_Q);
    std::cout << "[Decision] keyframe? " << (makeKF ? "YES" : "NO") << "\n";

    return 0;
}