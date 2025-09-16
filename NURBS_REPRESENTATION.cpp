// nurbs_representation.cpp
// g++ -std=c++17 nurbs_representation.cpp -O2 -I /usr/include/eigen3 -o nurbs
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

//====================  Lie(SO3)  ====================//
static inline Mat3 hat(const Vec3& w){
    Mat3 W; W <<    0, -w.z(),  w.y(),
                 w.z(),     0, -w.x(),
                -w.y(),  w.x(),     0;
    return W;
}

static inline Mat3 ExpSO3(const Vec3& w){
    double th = w.norm();
    if (th < 1e-12) return Mat3::Identity();
    Vec3 a = w/th;
    Mat3 A = hat(a);
    return Mat3::Identity() + std::sin(th)*A + (1-std::cos(th))*(A*A);
}

static inline Vec3 LogSO3(const Mat3& R){
    double cos_th = (R.trace()-1.0)*0.5;
    cos_th = std::min(1.0,std::max(-1.0,cos_th));
    double th = std::acos(cos_th);
    if (th < 1e-12) return Vec3::Zero();
    Vec3 w; 
    w << R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1);
    w *= 0.5/th;
    return w*th;
}


Eigen::Matrix4d buildBlendingM4(double m00, double m02, double m12,
                                double m22, double m32, double m33){
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    M(0,0) =  m00;
    M(0,1) =  1.0 - m00 - m02;
    M(0,2) =  m02;

    M(1,0) = -3.0*m00;
    M(1,1) =  3.0*m00 - m12;
    M(1,2) =  m12;

    M(2,0) =  3.0*m00;
    M(2,1) = -3.0*m00 - m22;
    M(2,2) =  m22;

    M(3,0) = -m00;
    M(3,1) =  m00 - m32;
    M(3,2) =  m32;
    M(3,3) =  m33;           // m33
}

//====================    ====================//
struct SegmentTime { double ti, ti1; }; // [ti, ti+1]
inline double normalized_u(double t, const SegmentTime& s){
    return (t - s.ti) / (s.ti1 - s.ti); // u \in [0,1]
}

Eigen::Vector3d lambda_vector(double t, const SegmentTime& seg,
                              const Eigen::Matrix4d& M4,
                              int i /*， M^(4)(i)；*/)
{
    // u_vec = [1, u, u^2, u^3]^T
    double u = normalized_u(t, seg);
    Eigen::Vector4d uvec(1.0, u, u*u, u*u*u);

    // λ = M^(4)(i) * u_vec，
    Eigen::Vector4d lam4 = M4 * uvec;
    return lam4.head<3>();
}

//====================   ====================//
struct CtrlPose {
    // 4 个控制点（i, i+1, i+2, i+3）
    Mat3 R[4];
    Vec3 x[4];
    
    double wbar[3] = {1.0/3, 1.0/3, 1.0/3};
};

struct Pose {
    Mat3 R;
    Vec3 x;
};

Pose evaluatePoseAt(double t, const SegmentTime& seg,
                    const Eigen::Matrix4d& M4,
                    const CtrlPose& cp)
{
    Eigen::Vector3d lam = lambda_vector(t, seg, M4, /*i=*/0);
    // R(t) = R_i Π_{j=1}^3 exp( wbar_j λ_j * log(R_{i+j-1}^{-1} R_{i+j}) )
    Mat3 R = cp.R[0];
    for(int j=1; j<=3; ++j){
        Mat3 Rrel = cp.R[j-1].transpose() * cp.R[j];
        Vec3 w = LogSO3(Rrel);
        R = R * ExpSO3(cp.wbar[j-1] * lam[j-1] * w);
    }
    // x(t) = x_i + Σ wbar_j λ_j * (x_{i+j} - x_{i+j-1})
    Vec3 x = cp.x[0];
    for(int j=1; j<=3; ++j){
        x += cp.wbar[j-1] * lam[j-1] * (cp.x[j] - cp.x[j-1]);
    }
    return {R, x};
}


double localDensity(double t1, double t2,
                    const std::vector<double>& keyTimes, double sigma)
{
    if (t2 <= t1) return 0.0;
    double c = 0.5*(t1 + t2);
    double inv = 1.0 / (t2 - t1);
    double sum = 0.0;
    for (double f : keyTimes){
        if (f >= t1 && f <= t2){
            double z = (f - c) / sigma;
            sum += std::exp(-0.5 * z*z);     // e^{-(f-c)^2 / (2σ^2)}
        }
    }
    return inv * sum;
}

//  D_g = F / (T2-T1)
inline double globalDensity(double T1, double T2, int F){
    if (T2 <= T1) return 0.0;
    return static_cast<double>(F) / (T2 - T1);
}

//  D = w_l D_l + w_g D_g
inline double combinedDensity(double Dl, double Dg,
                              double wl=0.8, double wg=0.2){
    return wl*Dl + wg*Dg;
}


// N = 1 + 3 * 1/(1 + e^{-k(D - D0)})
int estimateControlPointCount(double D,
                              double D0 = 12.0, double k = 0.8)
{
    double s = 1.0 + 3.0 / (1.0 + std::exp(-k*(D - D0)));
    int N = static_cast<int>(std::round(s));
    if (N < 1) N = 1;
    if (N > 4) N = 4;
    return N;
}

//====================  (12) （Lorentzian ） ====================//
// w(t) = Σ_{f_i∈[T1,T2]}  1 / ( 1 + ((t - f_i)/r)^2 )
double lorentzianWeight(double t, double T1, double T2,
                        const std::vector<double>& keyTimes, double r)
{
    double w = 0.0;
    for (double fi : keyTimes){
        if (fi >= T1 && fi <= T2){
            double z = (t - fi) / r;
            w += 1.0 / (1.0 + z*z);
        }
    }
    return w;
}

//====================   ====================//
int main(){
    // ---- 
    SegmentTime seg{0.0, 1.0};
    Eigen::Matrix4d M4 = buildBlendingM4(
        0.4, 0.2,   // m00, m02
        0.15, 0.15, // m12, m22
        0.1,  0.2   // m32, m33
    );


    CtrlPose cp;
    cp.R[0] = Mat3::Identity();
    cp.R[1] = ExpSO3(Vec3(0.0, 0.0, 0.3));
    cp.R[2] = ExpSO3(Vec3(0.0, 0.2, 0.6));
    cp.R[3] = ExpSO3(Vec3(0.1, 0.3, 0.9));
    cp.x[0] = Vec3(0,0,0);
    cp.x[1] = Vec3(1,0,0);
    cp.x[2] = Vec3(2,1,0);
    cp.x[3] = Vec3(3,1,1);
    cp.wbar[0] = 0.4; cp.wbar[1] = 0.35; cp.wbar[2] = 0.25;


    double t = 0.35;
    Pose p = evaluatePoseAt(t, seg, M4, cp);
    std::cout << "[Pose] x(t)=" << p.x.transpose() << "\n";


    std::vector<double> keyTimes{0.05,0.10,0.18,0.32,0.36,0.60,0.88};
    double Dl = localDensity(0.2, 0.5, keyTimes, /*sigma=*/0.06);
    double Dg = globalDensity(0.0, 1.0, static_cast<int>(keyTimes.size()));
    double D  = combinedDensity(Dl, Dg, /*wl=*/0.8, /*wg=*/0.2);
    int N     = estimateControlPointCount(D, /*D0=*/12.0, /*k=*/0.8);
    std::cout << "[Density] Dl="<<Dl<<" Dg="<<Dg<<" D="<<D<<" -> N="<<N<<"\n";

    double w = lorentzianWeight(0.35, 0.0, 1.0, keyTimes, /*r=*/0.08);
    std::cout << "[Weight] w(t=0.35) = " << w << "\n";

    return 0;
}
