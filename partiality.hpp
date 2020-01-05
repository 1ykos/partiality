#ifndef PARTIALITY_H
#define PARTIALITY_H

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

#include <dlib/matrix.h>

#include "wmath.hpp"
#include "geometry.hpp"
#include "patchmap.hpp"

namespace partiality{
  constexpr double pi          = 3.14159265358979323846;
  constexpr double LOG_DBL_MAX = 709.78271289338402993962517939506;
  using dlib::abs;
  using dlib::chol;
  using dlib::cholesky_decomposition;
  using dlib::diag;
  using dlib::diagm;
  using dlib::dot;
  using dlib::eigenvalue_decomposition;
  using dlib::identity_matrix;
  using dlib::inv;
  using dlib::inv_lower_triangular;
  using dlib::length;
  using dlib::length_squared;
  using dlib::make_symmetric;
  using dlib::matrix;
  using dlib::matrix_exp;
  using dlib::matrix_op;
  using dlib::normalize;
  using dlib::op_make_symmetric;
  using dlib::pinv;
  using dlib::round;
  using dlib::set_colm;
  using dlib::sqrt;
  using dlib::tmp;
  using dlib::trace;
  using dlib::zeros_matrix;
  using geometry::panel;
  using geometry::geometry;
  using std::abs;
  using std::accumulate;
  using std::array;
  using std::begin;
  using std::bind;
  using std::bitset;
  using std::cerr;
  using std::cin;
  using std::complex;
  using std::conditional;
  using std::cout;
  using std::end;
  using std::endl;
  using std::fill;
  using std::fixed;
  using std::floor;
  using std::function;
  using std::get;
  using std::getline;
  using std::ifstream;
  using std::invoke_result;
  using std::isnan;
  using std::istream;
  using std::make_tuple;
  using std::map;
  using std::max_element;
  using std::min;
  using std::minmax;
  using std::mt19937_64;
  using std::normal_distribution;
  using std::numeric_limits;
  using std::ofstream;
  using std::random_device;
  using std::exponential_distribution;
  using std::ref;
  using std::remainder;
  using std::round;
  using std::setprecision;
  using std::setw;
  using std::sqrt;
  using std::stod;
  using std::streamsize;
  using std::string;
  using std::stringstream;
  using std::swap;
  using std::tuple;
  using std::uniform_int_distribution;
  using std::uniform_real_distribution;
  using std::unordered_map;
  using std::unordered_set;
  using std::vector;
  using whash::patchmap;
  using wmath::bswap;
  using wmath::circadd;
  using wmath::clip;
  using wmath::digits;
  using wmath::inverf;
  using wmath::log2;
  using wmath::mean_variance;
  using wmath::mean_variance_M2_dw_acc;
  using wmath::mean_variance_M2_dw_ini;
  using wmath::mean_variance_M2_dx_acc;
  using wmath::mean_variance_M2_dx_ini;
  using wmath::mean_variance_mean_dw;
  using wmath::mean_variance_mean_dx;
  using wmath::mean_variance_sumw_dw;
  using wmath::mean_variance_sumw_dx;
  using wmath::popcount;
  using wmath::pow;
  using wmath::rol;

  using namespace std::placeholders;

  typedef tuple<int32_t,int32_t,int32_t> IDX;

  struct mosaicity_tensor{
    // rotation around a,b,c, scaling in a,b,c, arbitraryly correlated
    dlib::matrix<double,6,6> m;
  };

  /* Dk covariance contribution due to divergence
   * This models the incoming wave vector direction distribution.
   * It leads to a broadening of the observed peaks similar but discernable
   * from the reciprocal peak width.
   * deprecated
   */
  matrix<double,3,3> inline S_divergence(
      const matrix<double,3,1>& win, // incoming wave vector, |win| == 1
      const double& div,
      const double& wvn) {
    return pow(div*wvn,2u)*(identity_matrix<double>(3)-win*trans(win));
  }

  /* Dk covariance contribution due to dispersion aka bandwidth
   * given as the variance of the wavenumber i.e. σ(1/λ)
   * deprecated
   */
  matrix<double,3,3> inline S_bandwidth(
      const matrix<double,3,1>& v,
      const double& bnd) {
    return pow(bnd,2u)*v*trans(v);
  }

  /* reciprocal peak shape contribution due to inherent width
   * this makes the observed peaks broader, similar but discernable from the
   * influence of beam divergence.
   * This approximates the reciprocal peak shape by a spherical gaussian -
   * which is probably a poor approximation... please make up your own sum of
   * elliptical shapes if this contribution is significant. 
   */
  matrix<double,3,3> inline P_peakwidth(
      const double& rpw
      ) {
    const double rpw2 = pow(rpw,2u);
    return matrix<double,3,3>{rpw2,0,0,0,rpw2,0,0,0,rpw2};
  }
  
  /* reciprocal peak shape contribution due to mosaicity
   * This leads to a radial smearing of the reciprocal peaks orthogonal to
   * the hkl vector and to an radial smearing of the observed peaks.
   * On the detector this leads to a radial smearing of the peaks.
   */
  matrix<double,3,3> inline P_mosaicity(
      const matrix<double,3,1>& x,
      const double& mosaicity
      ) {
    const double m2 = mosaicity*mosaicity;
    const double nx2 = length_squared(x);
    return m2*(nx2*identity_matrix<double>(3)-x*trans(x));
  }

  matrix<double,3,1> inline cross_product(
      const matrix<double,3,1>& a,
      const matrix<double,3,1>& b) {
    return matrix<double,3,1>
    {a(1)*b(2)-a(2)*b(1),
     a(2)*b(0)-a(0)*b(2),
     a(0)*b(1)-a(1)*b(0)};
  }

  double solid_angle(
      const matrix<double,3,1> v0,
      const matrix<double,3,1> v1,
      const matrix<double,3,1> v2
      ) {
    const matrix<double,3,1> n0 = normalize(v0);
    const matrix<double,3,1> n1 = normalize(v1);
    const matrix<double,3,1> n2 = normalize(v2);
    return length(cross_product(n1-n0,n2-n0));
  }
  
  matrix<double,3,3> inline P_rotation(
      const matrix<double,3,1>& x,
      const double& a,               // angle
      const matrix<double,3,1>& n    // axis
      ) {
    const matrix<double,3,1> c = a*cross_product(x,n);
    return c*trans(c);
  }
  
  /* strain parametrizes the variance of the distribution of different strain
   * in the unit cells of the crystal. This leads to radially elongation of the
   * reciprocal peaks and consequently to an elongation of the observed,
   * virtually indistinguishable from the elongation due to wavelength bandwith.
   * The two parameters are degenerate for all practical purposes.
   */
  matrix<double,3,3> inline P_strain(
      const matrix<double,3,1> x,
      const double& strain
      ) {
    return (strain*x)*trans(strain*x);
  }
  
  // rotate two normalized vectors onto another
  matrix<double,3,3> const rotation_matrix(
      const matrix<double,3,1>& v0,
      const matrix<double,3,1>& v1
      ) {
    const matrix<double,3,1> x = normalize(cross_product(v0,v1));
    const double cos_a = trans(v0)*v1;
    const double sin_acos_cos_a = sqrt(1-cos_a*cos_a);
    const matrix<double,3,3> A
    { 0   , -x(2), x(1),
      x(2),  0   ,-x(0),
     -x(1),  x(0), 0    };
    return identity_matrix<double>(3)+sin_acos_cos_a*A+(1-cos_a)*A*A;
  }
  
  // rotate two normalized vectors onto another by mirroring twice
  // not tested
  matrix<double,3,3> const rotation_matrix_v1(
      const matrix<double,3,1>& v0,
      const matrix<double,3,1>& v1
      ) {
    const matrix<double,3,1> v2 = normalize(v0+v1);
    const matrix<double,3,3> M0 =
      identity_matrix<double>(3)-v1*trans(v1);
    const matrix<double,3,3> M1 =
      identity_matrix<double>(3)-v2*trans(v2);
    return M1*M0;
  }

  matrix<double,3,3> const rotation_matrix(
      const double a,
      const matrix<double,3,1>& u
      ) {
    const double q0 = u(0)*sin(a/2);
    const double q1 = u(1)*sin(a/2);
    const double q2 = u(2)*sin(a/2);
    const double q3 = cos(a/2);
    return matrix<double,3,3>
    {
      q0*q0+q1*q1-q2*q2-q3*q3,         2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2),
              2*(q1*q2+q0*q3), q0*q0-q1*q1+q2*q2-q3*q3,         2*(q2*q3-q0*q1),
              2*(q1*q3-q0*q2),         2*(q2*q3+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3
    };
  }

  struct source{
    double flx = 1.0;        // photon flux
    matrix<double,3,1> kin;  // incoming wave vector
    matrix<double,3,3> S12;  // divergence and bandwidth *not squared*
    matrix<double,3,3> S12w(const matrix<double,3,1>& w) const {
      const matrix<double,3,1> win   = normalize(kin);
      const matrix<double,3,3> R     = rotation_matrix(win,w);
      // isolate contribution in direction of wave vector aka bandwidth
      const matrix<double,3,3> S12_b = double(trans(win)*S12*win)*win*trans(win);
      // get covariance matrix of correlated difference
      const matrix<double,3,3> S12_w = S12 - R * S12_b;
      // cerr << trans(w) << trans(R*win);
      //cerr << "testing rotation" << endl;
      //cerr << S12_w*trans(S12_w) << endl;
      //cerr << 0.03*0.03*(w-win)*trans(w-win) << endl;
      return S12_w;
    }
    matrix<double,3,3> Sw(const matrix<double,3,1>& w) const {
      const matrix<double,3,3> S12_w = S12w(w);
      return S12_w*trans(S12_w);
    }
    // area of ewald sphere without width
    const double ewald_width(const matrix<double,3,1>& w) const {
      return trans(w)*S12w(w)*w;
    }
    const double area() const {
      return 4*pi*length_squared(kin);
    }
    // volume of ewald sphere of given width 
    const double volume(const matrix<double,3,1>& w) const {
      return sqrt(2*pi)*ewald_width(w)*area();
    }
  };

  struct crystl{
    matrix<double,3,3> U = zeros_matrix<double>(3,3);   // unit cell
    matrix<double,3,3> R = zeros_matrix<double>(3,3);   // reciprocal unit cell 
    double mosaicity = 0;                               // mosaicity
    matrix<double,3,3> peak = zeros_matrix<double>(3,3);// reciprocal peak
                                                        // *squared*
    double strain=0;                                    // crystal strain
    // scaling parameters a exp( - 0.5 b q² )
    double a=1.0,b=0.0;
    double scaling(const matrix<double,3,1>& x) const {
      return a*exp(-0.5*b*length_squared(x));
    }
    matrix<double,3,1> scaling_dx(const matrix<double,3,1>& x) const {
      return -b*scaling(x)*x;
    }
    double scaling_da(const matrix<double,3,1>& x) const {
      return exp(-0.5*b*length_squared(x));
    }
    double scaling_db(const matrix<double,3,1>& x) const {
      return -0.5*length_squared(x)*scaling(x);
    }
    matrix<double,3,3> S_mosaicity(const matrix<double,3,1>& x ) const {
      const double m2 = mosaicity*mosaicity;
      const double nx2 = length_squared(x);
      return m2*(nx2*identity_matrix<double>(3)-x*trans(x));
    }
    matrix<double,3,3> inline S_strain(const matrix<double,3,1> x ) const {
      const double s2 = strain*strain;
      return s2*x*trans(x);
    }
    matrix<double,3,3> S(const matrix<double,3,1>& x) const {
      return peak + S_mosaicity(x) + S_strain(x);
    }
    crystl& operator+=(const crystl& o) {
      U         += o.U;
      R         += o.R;
      mosaicity += o.mosaicity;
      peak      += o.peak;
      strain    += o.strain;
      a         += o.a;
      b         += o.b;
      return *this;
    }
    crystl& operator-=(const crystl& o) {
      U         -= o.U;
      R         -= o.R;
      mosaicity -= o.mosaicity;
      peak      -= o.peak;
      strain    -= o.strain;
      a         -= o.a;
      b         -= o.b;
      return *this;
    }
    crystl operator+(const crystl& o) const {
      struct crystl result = *this;
      return (result+=o);
    }
    crystl operator-(const crystl& o) const {
      struct crystl result = *this;
      return (result-=o);
    }
    crystl& operator*=(const double c) {
      U         *= c;
      R         *= c;
      mosaicity *= c;
      peak      *= c;
      strain    *= c;
      a         *= c;
      b         *= c;
      return *this;
    }
    crystl operator*(const double c) const {
      struct crystl result = *this;
      return (result*=c);
    }
    crystl operator/(const double c) const {
      const double d = 1.0/c;
      return (*this)*(1.0/c);
    }
  };

  source const inline average_source(
      const vector<source>& sources
      ) {
    double sumw = 0;
    matrix<double,3,1> mean{0,0,0};
    matrix<double,3,3> M2{0,0,0,0,0,0,0,0,0};
    for (const source& source : sources) {
      const double temp = source.flx + sumw;
      const matrix<double,3,1> delta = source.kin - mean;
      const matrix<double,3,1> R     = delta*source.flx/temp;
      mean += R;
      M2   += delta * trans(delta) * sumw * source.flx / temp
           +  source.flx * ( source.S12 * trans(source.S12) );
      sumw = temp;
    }
    M2/=sumw;
    // now calculate square root
    eigenvalue_decomposition<matrix<double,3,3>> evd(make_symmetric(M2));
    const matrix<double,3,3> V    = evd.get_pseudo_v();
    const matrix<double,3,1> D12v = sqrt(evd.get_real_eigenvalues());
    const matrix<double,3,3> S12  = V * diagm(D12v) * inv(V);
    return {sumw,mean,S12};
  }
  /*
  matrix<double,3,3> const inline constant_covariance_matrix(
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const source& source,
      const crystl& crystl
      ) {
    return pow(source.wvn,2u)*source.div
          +pow(source.bnd,2u)*identity_matrix<double>(3) // isotropic
          +P_mosaicity(m,crystl.mosaicity)
          +P_strain(m,crystl.strain)
          +crystl.peak;
  }
  */
  /*
  // covariance contribution of source
  matrix<double,3,3> const inline matrix_S(
      const matrix<double,3,1>& Dw,
      const source& source
      ){
    return pow(source.wvn,2u)*source.div
          +S_bandwidth(Dw,source.bnd);
  }
  */
  /*
  // covariance contriubtion of crystal
  matrix<double,3,3> const inline matrix_P(
      const matrix<double,3,1>& m,
      const crystl& crystl
      ){
    return P_mosaicity(m,crystl.mosaicity)
          +P_strain(m,crystl.strain)
          +crystl.peak;
  }
  */
  /*
  matrix<double,3,3> const inline covariance_matrix( // Dk covariance
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const struct source& source,
      const struct crystl& crystl
      ){
    return  matrix_S(Dw,source)+matrix_P(m,crystl);
  }
  */
  /*
  template<size_t verbosity=0>
  matrix<double,3,1> const inline optimize_wout(
      const matrix<double,3,1>& w,
      const matrix<double,3,1>& m,
      const struct source& source,
      const struct crystl& crystl,
      bool& success
      ){
    if constexpr (verbosity>3) cerr << "optimize_wout" << endl;
    const double wvn = source.wvn;
    const matrix<double,3,1> win= source.win;
    const matrix<double,3,3> S  = covariance_matrix(w-win,m,source,crystl);
    const matrix<double,3,3> iS = inv(S);
    const matrix<double,3,3> wtw= w*trans(w);
    const matrix<double,3,3> P  = identity_matrix<double>(3)-wtw;
    const matrix<double,3,1> m1 = m/wvn+win;
    const matrix<double,3,1> d  = w-m1;
    const matrix<double,3,3> iPiSPpwtw = pinv(P*iS*P+wtw);
    const matrix<double,3,1> Dx = iPiSPpwtw*P*iS*d;
    double v = pow(wvn,2u)*trans(w-m1)*iS*(w-m1);
    if constexpr (verbosity>3) cerr << v << endl;
    for (double c=1.0;c>1e-4;c*=exp(-1)){
      if constexpr (verbosity>3) cerr << "search loop" << endl;
      const matrix<double,3,1> w1=normalize(w-c*Dx);
      const matrix<double,3,3> iS=inv(covariance_matrix(w1-win,m,source,crystl));
      const double _v = pow(wvn,2u)*trans(w1-m1)*iS*(w1-m1);
      if constexpr (verbosity>3) cerr << _v << " " << v << endl;
      if constexpr (verbosity>3) cerr << trans(w1);
      if (pow(wvn,2u)*trans(w1-m1)*iS*(w1-m1)<v) return w1;
    }
    if constexpr (verbosity>3) cerr << "not a minimizing direction :(" << endl;
    success = false;
    return w;
  }
  */

  enum class predict_mode : size_t {
    candidate_hkls,
    pixel_partiality,
    index_partiality
  };
 
  template<class matrix_exp>
  const auto trace(const matrix_exp& m1) {
    assert(m1.nc()==m1.nr());
    matrix<typename matrix_exp::value_type,1,1> result =
      zeros_matrix<typename matrix_exp::value_type>(1,1);
    for (size_t i=0;i!=min(m1.nc(),m1.nr());++i) result(1,1) += m1(i,i);
    return result;
  }

  auto const inline predict(
      const matrix<double,3,1> w,   // outgoing wave direction
      const matrix<double,3,3> psf, // sampling correction
      const IDX& hkl,
      const source& source,
      const crystl& crystl
      )
  {
    const matrix<double,3,1> dhkl
    {
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)
    };
    const matrix<double,3,1> x  = crystl.R*dhkl;
    const matrix<double,3,3> P  = crystl.peak;
    const matrix<double,3,1> wi = normalize(source.kin);
    const matrix<double,3,3> Sm = crystl.S_mosaicity(x);
    const matrix<double,3,3> Ss = crystl.S_strain(x);
    const matrix<double,3,3> S0 = P+Sm+Ss;
    // compute the q - vector from point on the detector
    // p is the projection of w and y onto the ewald sphere
    const matrix<double,3,1> p = w*length(source.kin)-source.kin;
    // S1 is the total covariance of the source in reciprocal space in
    // direction w
    const matrix<double,3,3> Sw      = source.Sw(w);
    const matrix<double,3,3> S1      = Sw + psf;
    const matrix<double,3,3>  S_circ = S0+S1;
    const matrix<double,3,3> iS_circ = inv(S_circ);
    const matrix<double,3,3> iS0     = inv(S0);
    const matrix<double,3,3> iS1     = inv(S1);
    // S_stari is the inverse of S_star
    const matrix<double,3,3> iS_star = iS0+iS1;
    const matrix<double,3,3>  S_star = inv(iS_star);
    const double scaling             = crystl.scaling(x);
    const double nlog_exponential    = 0.5*trans(x-p)*iS_circ*(x-p);
    const double exponential         = //(nlog_exponential<LOG_DBL_MAX)?
                                       exp(-nlog_exponential);
                                     //: 0.0;
    // normalisation, note the additional normalisation to
    // trans(wi)*P*wi to decorrelate P and linear scaling
    const double normalisation_p0    = sqrt(trans(wi)*P*wi);
    const double normalisation_p1    = 1/(2*pi*sqrt(det(S_circ))*source.area());
              // 1/(2*pi*(det(S+P)/((wi)'*P*wi))^(1/2)*sa)
    const double normalisation       = normalisation_p0*normalisation_p1;
              // 1/(2*pi*sqrt(det(S_circ)/double(trans(wi)*P*wi))*source.area());
    const double flx                 = scaling*exponential*normalisation;
    struct crystl dflx;
    const double tr_iS_circ = iS_circ(0,0)+iS_circ(1,1)+iS_circ(2,2); 
    matrix<double,3,3> dexp_dR;
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const double             t2 = pow(crystl.strain,2);
      const matrix<double,3,1> t3 = x-p;
      const double             t4 = pow(crystl.mosaicity,2);
      const matrix<double,3,3> T5 = iS_circ;
      const matrix<double,3,1> t6 = T5*t3;
      const matrix<double,3,3> T7 = t6*trans(dhkl);
      const double             t8 = trans(t3)*T5*t0;
      dexp_dR = -T7+t2*t8*T7+t4*double(trans(t3)*T5*t6)*t0*trans(dhkl)-t4*t8*T7;
      dexp_dR*= exponential;
    }
    matrix<double,3,3> dnorm_dR;
    {
      const double             t0 = pow(crystl.strain,2);
      const matrix<double,3,1> t1 = x;
      const matrix<double,3,3> T2 = x*trans(x);
      const double             t3 = 0.5;
      const double             t4 = trans(wi)*P*wi;
      const double             t5 = pow(crystl.mosaicity,2);
      const matrix<double,3,3> T6 = S_circ;
      const double             t7 = det(S_circ);
      const double             t8 = 1.0/sqrt(t7);
      const double             t9 = 1.0/sqrt(t4);
      const double             t10= t4*t8*t9;
      const double             t11= pi*t7;
      const matrix<double,3,3> T12= det(S_circ)*iS_circ;
      const double             t13= 4*source.area()*t11;
      const matrix<double,3,3> T14= T12*x*trans(dhkl);
      dnorm_dR =
        - 2*t0*t10/t13*T14
        - t10*t5/(t11*source.area()*2)*(T12(0,0)+T12(1,1)+T12(2,2))*t1*trans(dhkl)
        + 2*t4*t5*t8*t9/t13*T14;
    }
    dflx.R = crystl.scaling_dx(x)*trans(dhkl)*exponential*normalisation
            +scaling*dexp_dR*normalisation
            +scaling*exponential*dnorm_dR;
    double dexp_dmosaicity;
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 = 
        length_squared(x)*identity_matrix<double>(3)-x*trans(x);
      const matrix<double,3,3> T3 = iS_circ;
      const matrix<double,3,1> t4 = x-p;
      dexp_dmosaicity = crystl.mosaicity*trans(t4)*T3*T2*T3*t4;
      dexp_dmosaicity*=exponential;
    }
    double dnorm_dmosaicity;
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const double             t2 = 0.5;
      const double             t3 = trans(wi)*P*wi;
      const matrix<double,3,3> T4 =
        length_squared(x)*identity_matrix<double>(3)-x*trans(x);
      const matrix<double,3,3> T5 = S_circ;
      const double             t6 = det(S_circ);
      const matrix<double,3,3> T7 = T4*det(T5)*iS_circ;
      dnorm_dmosaicity =
        - crystl.mosaicity*t3/sqrt(t6*t3)*(T7(0,0)+T7(1,1)+T7(2,2))
        / (t6*pi*source.area()*2);
    }
    dflx.mosaicity =
        scaling*dexp_dmosaicity*normalisation
      + scaling*exponential*dnorm_dmosaicity;
/*      - crystl.scaling(x)
      * (
          // dexp
          exp(-0.5*trans(x-p)*S_circi*(x-p))
            * (
               crystl.mosaicity
             * trans(x-p)
             * S_circi
             *  * S_circi * (x-p)
              )
            /(2*pi*sqrt(det(S_circ)/(trans(wi)*P*wi))*source.area());
          // dnorm
         +crystl.mosaicity * det(S_circ)^(-3/2) * ( trans(wi) * P * wi )
          * tr( (length_squared(x)*I-x*trans(x)) * det(S_circ)*S_circi ) 
        )
      ;*/
    double dexp_dstrain;
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,1> t2 = x-p;
      const matrix<double,3,3> T3 = iS_circ;
      dexp_dstrain = crystl.strain*double(trans(x)*T3*t2)*double(trans(t2)*T3*t0);
      dexp_dstrain*= exponential;
    }
    double dnorm_dstrain;
    {
      const double t3 = trans(wi)*P*wi;
      const double t5 = det(S_circ);
      dnorm_dstrain =
        - crystl.strain*t3/sqrt(t5*t3)*double(trans(x)*det(S_circ)*iS_circ*x)
        / (2*t5*pi*source.area());
    }
    dflx.strain =
        scaling*dexp_dstrain*normalisation
      + scaling*exponential*dnorm_dstrain;
    const matrix<double,3,3> dexp_dpeak =
      exponential*iS_circ*(x-p)*trans(x-p)*iS_circ/2; 
    matrix<double,3,3> dnorm_dpeak;
    /*{
      // 1/(2*pi*(det(S+P)/((wi)'*P*wi))^(1/2)*sa)
      const double             t0 = 0.5;
      const double             t1 = trans(wi)*P*wi;
      const matrix<double,3,3> T2 = S_circ;
      const double             t3 = det(S_circ);
      const double             t4 = trans(wi)*trans(P)*wi;
      const double             t5 = 4*pi*source.area()*t3;
      dnormp1_dpeak =
        - (trans((t1*pow(t3,t0-1)*pow(t1,-t0))/t5*det(T2)*iS_circ)
          -  (t4*pow(t4,-(1+t0))*pow(t3,t0))*wi*trans(wi));
    }*/
    {
      const matrix<double,3,3> dnormp1_dpeak =
        - trans(iS_circ/(4*pi*source.area()*sqrt(det(S_circ)))); 
      const matrix<double,3,3> dnormp0_dpeak =
        0.5*wi*trans(wi)/sqrt(trans(wi)*trans(P)*wi);
      dnorm_dpeak =
        normalisation_p0*dnormp1_dpeak+normalisation_p1*dnormp0_dpeak;
    }
    dflx.peak = 
        scaling*dexp_dpeak*normalisation
      + scaling*exponential*dnorm_dpeak;
    dflx.a = crystl.scaling_da(x)*exponential*normalisation;
    dflx.b = crystl.scaling_db(x)*exponential*normalisation;
    const matrix<double,3,1> dw = w-normalize(source.kin);
    const matrix<double,3,1> ww = dw/length_squared(dw);
    const double wvn = trans(ww)*S_star*(iS0*x+iS1*p);
    struct crystl dwvn;
    {
      const double             t0 = pow(crystl.strain,2);
      const matrix<double,3,1> t1 = x;
      const matrix<double,3,3> T2 = x*trans(x);
      const double             t3 = pow(crystl.mosaicity,2);
      const matrix<double,3,3> T4 = iS0;
      const matrix<double,3,3> T5 = iS1;
      const matrix<double,3,3> T6 = S_star;
      const matrix<double,3,1> t7 = T4*x;
      const matrix<double,3,1> t8 = T6*t7;
      const matrix<double,1,3> t9 = trans(x)*T4+trans(p)*T5;
      const matrix<double,3,1> t10= T4*T6*ww;
      const double             t11= t9*t8;
      const matrix<double,3,3> T12= t10*trans(dhkl);
      const double             t13= trans(ww)*t8;
      const matrix<double,3,3> T14= T4*T6*(t7+T5*p)*trans(dhkl);
      const double             t15= t0*t13;
      const double             t16= 2*t3;
      const matrix<double,3,1> t17= T4*t10;
      const matrix<double,3,3> T18= t1*trans(dhkl);
      const double             t19= trans(x)*t7;
      const double             t20= t13*t3;
      const matrix<double,3,3> T21= t7*trans(dhkl);
      dwvn.R =
        T12
      + t0*t11*T12
      + t15*T14
      + t16*double(t9*T6*t17)*T18
      - t11*t3*T12
      - t20*T14
      - t0*t19*T12
      - t15*T21
      - t16*double(trans(x)*t17)*T18
      + t19*t3*T12
      + t20*T21;
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x); 
      const matrix<double,3,3> T2 =
        length_squared(x)*identity_matrix<double>(3)-x*trans(x); 
      const matrix<double,3,3> T3 = iS0;
      const matrix<double,3,3> T4 = iS1;
      const matrix<double,3,3> T5 = S_star;
      const double             t6 = 2*crystl.mosaicity;
      const matrix<double,3,1> t7 = T3*x;
      const double             t8 = length_squared(dw);

      dwvn.mosaicity =
         t6*trans(dw)*T5*T3*T2*T3*T5*(t7+T4*p)/t8
       - t6*trans(dw)*T5*T3*T2*t7/t8;
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 = iS0;
      const matrix<double,3,3> T3 = iS1;
      const matrix<double,3,3> T4 = S_star;
      const double             t5 = 1.0/length_squared(dw);
      const matrix<double,3,1> t6 = T2*T4*dw;
      const matrix<double,1,3> t7 = trans(x)*T2;
      dwvn.peak =
        t5*t6*((t7+trans(p)*T3)*T4*T2)-t5*t6*t7;
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 = iS0;
      const matrix<double,3,3> T3 = iS1;
      const matrix<double,3,3> T4 = S_star;
      const matrix<double,3,1> t5 = T2*t0;
      const double             t6 = 2*crystl.strain;
      const double             t7 = trans(dw)*T4*t5;
      const double             t8 = length_squared(dw);
      dwvn.strain =
        (t6*t7*trans(x)*T2*T4*(t5+T3*p))/t8
       -(t6*t7*trans(x)*t5)/t8;
    }
    dwvn.a = 0;
    dwvn.b = 0;
    const double bnd = trans(ww)*S_star*ww;
    struct crystl dbnd;
    {
      const double             t0 = pow(crystl.strain,2);
      const matrix<double,3,1> t1 = x;
      const matrix<double,3,3> T2 = x*trans(x);
      const double             t3 = pow(crystl.mosaicity,2);
      const matrix<double,3,3> T4 = iS0;
      const matrix<double,3,3> T5 = S_star;
      const matrix<double,3,1> t6 = T4*T5*ww;
      const double             t7 = 2*t3;
      const double             t8 = trans(ww)*T5*T4*t1;
      const matrix<double,3,3> T9 = t6*trans(dhkl);
      dbnd.R =
          2*t0*t8*T9
        + t7*double(trans(ww)*T5*T4*t6)*t1*trans(dhkl)
        - t7*t8*T9;
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 =
        length_squared(x)*identity_matrix<double>(3)-x*trans(x);
      const matrix<double,3,3> T3 = iS0;
      const matrix<double,3,3> T4 = S_star;
      dbnd.mosaicity = 2*crystl.mosaicity*trans(ww)*T4*T3*T2*T3*T4*ww;
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 = iS0;
      const matrix<double,3,3> T3 = S_star;
      dbnd.peak = T2*T3*ww*(trans(ww)*T3*T2);
    }
    {
      const matrix<double,3,1> t0 = x;
      const matrix<double,3,3> T1 = x*trans(x);
      const matrix<double,3,3> T2 = iS0;
      const matrix<double,3,3> T3 = S_star;
      dbnd.strain = 2*crystl.strain*trans(x)*T2*T3*ww*trans(ww)*T3*T2*x;
    }
    dbnd.a = 0;
    dbnd.b = 0;
    dflx.U = -trans(crystl.R)*dflx.R*trans(crystl.R);
    dwvn.U = -trans(crystl.R)*dwvn.R*trans(crystl.R);
    dbnd.U = -trans(crystl.R)*dbnd.R*trans(crystl.R);
    return make_tuple(flx,wvn,bnd,dflx,dwvn,dbnd);
  }

  void test_predict(){
    struct source source;
    source.flx = 7.0;
    source.kin(0) = 0;
    source.kin(1) = 0;
    source.kin(2) = 101;
    source.S12 = zeros_matrix<double>(3,3);
    source.S12(0,0) = 1e-4;
    source.S12(1,1) = 1e-4;
    source.S12(2,2) = 1;
    random_device rd;
    uniform_real_distribution<double> udistr(-2,2);
    uniform_int_distribution<int32_t> idistr(-5,5);
    exponential_distribution<double> edistr;
    struct crystl crystl;
    crystl.R = zeros_matrix<double>(3,3);
    while ((det(crystl.R)<0.25)||(det(crystl.R)>1))
      for (size_t i=0;i!=3;++i)
        for (size_t j=0;j!=3;++j)
          crystl.R(i,j) = udistr(rd);
    crystl.U = inv(crystl.R);
    crystl.mosaicity = udistr(rd)*udistr(rd)*udistr(rd)*udistr(rd);
    for (size_t i=0;i!=3;++i)
      for (size_t j=0;j!=3;++j)
        crystl.peak(i,j) = udistr(rd)*1e-2;
    crystl.peak=crystl.peak*trans(crystl.peak);
    crystl.strain = edistr(rd)*1e-3;
    crystl.a = 1+edistr(rd)*16;
    crystl.b = udistr(rd)*1e-2;
    IDX idx;
    matrix<double,3,1> max_y;
    double max_flx = 0;
    for (size_t i=0;i!=1u<<16;++i) {
      const matrix<double,3,1> w =
        normalize(matrix<double,3,1>{udistr(rd),udistr(rd),4+udistr(rd)});
      const matrix<double,3,3> psf =
        (identity_matrix<double>(3)-w*trans(w))/pow(32,2);
      IDX _idx{idistr(rd),idistr(rd),idistr(rd)};
      while (!get<0>(_idx)) get<0>(_idx) = idistr(rd);
      while (!get<1>(_idx)) get<1>(_idx) = idistr(rd);
      while (!get<2>(_idx)) get<2>(_idx) = idistr(rd);
      const auto result = predict(w,psf,_idx,source,crystl);
      if (get<0>(result)>=max_flx) {
        max_flx = get<0>(result);
        max_y = 512*w;
        idx = _idx; 
      } else if (get<0>(result)<0) {
        cerr << "predicting negative flux, this must not happen" << endl;
        exit(1);
      }
    }
    for (int i=-16;i<=16;++i) {
      for (int j=-16;j<=16;++j) {
        matrix<double,3,1> y = max_y;
        y(0)+=i;
        y(1)+=j;
        const matrix<double,3,1> w = normalize(y);
        const matrix<double,3,3> psf =
          (identity_matrix<double>(3)-w*trans(w))*pow(512,-2);
        const auto prediction = predict(w,psf,idx,source,crystl);
        if (get<0>(prediction)<1e-20) continue;
        cout << setw(16) << get<0>(prediction) << " "
             << setw(15) << get<1>(prediction) << " "
             << setw(15) << get<2>(prediction) << endl;
        struct crystl dflx;
        struct crystl dwvn;
        struct crystl dbnd;
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j) {
          const double  eps = 1e-8;
          struct crystl plus  = crystl;
          plus.R(i,j)+=eps;
          plus.U = inv(plus.R);
          struct crystl minu = crystl;
          minu.R(i,j)-=eps;
          minu.U = inv(minu.R);
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.R(i,j) = (get<0>(pv)-get<0>(mv))/(plus.R(i,j)-minu.R(i,j));
          dwvn.R(i,j) = (get<1>(pv)-get<1>(mv))/(plus.R(i,j)-minu.R(i,j));
          dbnd.R(i,j) = (get<2>(pv)-get<2>(mv))/(plus.R(i,j)-minu.R(i,j));
        }
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j) {
          const double  eps = 1e-8;
          struct crystl plus  = crystl;
          plus.U(i,j)+=eps;
          plus.R = inv(plus.U);
          struct crystl minu = crystl;
          minu.U(i,j)-=eps;
          minu.R = inv(minu.U);
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.U(i,j) = (get<0>(pv)-get<0>(mv))/(plus.U(i,j)-minu.U(i,j));
          dwvn.U(i,j) = (get<1>(pv)-get<1>(mv))/(plus.U(i,j)-minu.U(i,j));
          dbnd.U(i,j) = (get<2>(pv)-get<2>(mv))/(plus.U(i,j)-minu.U(i,j));
        }
        {
          const double eps = 1e-4;
          struct crystl plus = crystl;
          plus.mosaicity*=1+eps;
          struct crystl minu = crystl;
          minu.mosaicity*=1-eps;
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.mosaicity = (get<0>(pv)-get<0>(mv))
            /(plus.mosaicity-minu.mosaicity);
          dwvn.mosaicity = (get<1>(pv)-get<1>(mv))
            /(plus.mosaicity-minu.mosaicity);
          dbnd.mosaicity = (get<2>(pv)-get<2>(mv))
            /(plus.mosaicity-minu.mosaicity);
        }
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j) {
          const double eps = 1e-8;
          struct crystl plus = crystl;
          plus.peak(i,j)+=eps;
          struct crystl minu = crystl;
          minu.peak(i,j)-=eps;
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.peak(i,j) = (get<0>(pv)-get<0>(mv))
                          /(plus.peak(i,j)-minu.peak(i,j));
          dwvn.peak(i,j) = (get<1>(pv)-get<1>(mv))
                          /(plus.peak(i,j)-minu.peak(i,j));
          dbnd.peak(i,j) = (get<2>(pv)-get<2>(mv))
                          /(plus.peak(i,j)-minu.peak(i,j));
        }
        {
          const double eps = 1e-6;
          struct crystl plus = crystl;
          plus.strain*=1+eps;
          struct crystl minu = crystl;
          minu.strain*=1-eps;
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.strain = (get<0>(pv)-get<0>(mv))
            /(plus.strain-minu.strain);
          dwvn.strain = (get<1>(pv)-get<1>(mv))
            /(plus.strain-minu.strain);
          dbnd.strain = (get<2>(pv)-get<2>(mv))
            /(plus.strain-minu.strain);
        }
        {
          const double eps = 1e-6;
          struct crystl plus = crystl;
          plus.a*=1+eps;
          struct crystl minu = crystl;
          minu.a*=1-eps;
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.a = (get<0>(pv)-get<0>(mv))/(plus.a-minu.a);
          dwvn.a = (get<1>(pv)-get<1>(mv))/(plus.a-minu.a);
          dbnd.a = (get<2>(pv)-get<2>(mv))/(plus.a-minu.a);
        }
        {
          const double  eps = 1e-8;
          struct crystl plus = crystl;
          plus.b+=eps;
          struct crystl minu = crystl;
          minu.b-=eps;
          const auto pv = predict(w,psf,idx,source,plus);
          const auto mv = predict(w,psf,idx,source,minu);
          dflx.b = (get<0>(pv)-get<0>(mv))/(plus.b-minu.b);
          dwvn.b = (get<1>(pv)-get<1>(mv))/(plus.b-minu.b);
          dbnd.b = (get<2>(pv)-get<2>(mv))/(plus.b-minu.b);
        }
        cerr << "dR" << endl;
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j)
          cerr << setw(16) << get<3>(prediction).R(i,j) << " "
               << setw(15) << get<4>(prediction).R(i,j) << " "
               << setw(15) << get<5>(prediction).R(i,j) << " "
               << setw(15) << dflx.R(i,j) << " "
               << setw(15) << dwvn.R(i,j) << " "
               << setw(15) << dbnd.R(i,j) << endl;
        cerr << "dU" << endl;
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j)
          cerr << setw(16) << get<3>(prediction).U(i,j) << " "
               << setw(15) << get<4>(prediction).U(i,j) << " "
               << setw(15) << get<5>(prediction).U(i,j) << " "
               << setw(15) << dflx.U(i,j) << " "
               << setw(15) << dwvn.U(i,j) << " "
               << setw(15) << dbnd.U(i,j) << endl;
        cerr << "dmosaicity" << endl;
        cerr << setw(16) << get<3>(prediction).mosaicity << " "
             << setw(15) << get<4>(prediction).mosaicity << " "
             << setw(15) << get<5>(prediction).mosaicity << " "
             << setw(15) << dflx.mosaicity << " "
             << setw(15) << dwvn.mosaicity << " "
             << setw(15) << dbnd.mosaicity << endl;
        cerr << "dpeak" << endl;
        for (size_t i=0;i!=3;++i) for (size_t j=0;j!=3;++j)
          cerr << setw(16) << get<3>(prediction).peak(i,j) << " "
               << setw(15) << get<4>(prediction).peak(i,j) << " "
               << setw(15) << get<5>(prediction).peak(i,j) << " "
               << setw(15) << dflx.peak(i,j) << " "
               << setw(15) << dwvn.peak(i,j) << " "
               << setw(15) << dbnd.peak(i,j) << endl;
        cerr << "dstrain" << endl;
        cerr << setw(16) << get<3>(prediction).strain << " "
             << setw(15) << get<4>(prediction).strain << " "
             << setw(15) << get<5>(prediction).strain << " "
             << setw(15) << dflx.strain << " "
             << setw(15) << dwvn.strain << " "
             << setw(15) << dbnd.strain << endl;
        cerr << "da" << endl;
        cerr << setw(16) << get<3>(prediction).a << " "
             << setw(15) << get<4>(prediction).a << " "
             << setw(15) << get<5>(prediction).a << " "
             << setw(15) << dflx.a << " "
             << setw(15) << dwvn.a << " "
             << setw(15) << dbnd.a << endl;
        cerr << "db" << endl;
        cerr << setw(16) << get<3>(prediction).b << " "
             << setw(15) << get<4>(prediction).b << " "
             << setw(15) << get<5>(prediction).b << " "
             << setw(15) << dflx.b << " "
             << setw(15) << dwvn.b << " "
             << setw(15) << dbnd.b << endl;
      }
    }
  }

  const double wout_target(
      const source& source,
      const crystl& crystl,
      const IDX& hkl,
      const matrix<double,3,1> _w
      ){
    const matrix<double,3,1> dhkl
    {
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)
    };
    const matrix<double,3,1> x   = crystl.R*dhkl;
    const matrix<double,3,1> w   = normalize(_w);
    const double             wvn = length(source.kin);
    const matrix<double,3,1> ko  = wvn*w;
    const matrix<double,3,1> p   = ko-source.kin;
    const matrix<double,3,3> iS  = inv(crystl.S(x)+source.Sw(w));
    return trans(x-p)*inv(crystl.S(x)+source.Sw(w))*(x-p);
  }
  
  matrix<double,3,1> optimize_wout(
      const source& source,
      const crystl& crystl,
      const IDX& hkl,
      const matrix<double,3,1> w0){
    const matrix<double,3,1> dhkl
    {
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)
    };
    const matrix<double,3,1> x   = crystl.R*dhkl;
    const matrix<double,3,1> w   = normalize(w0);
    const double             wvn = length(source.kin);
    const matrix<double,3,1> ko  = wvn*w;
    const matrix<double,3,1> p   = ko-source.kin;
    const matrix<double,3,3> S   =
        crystl.S(x)
      + source.Sw(w);
    const matrix<double,3,3> iS= inv(S);
    const matrix<double,3,1> dw= -2*(wvn*iS*(x-p)-double(trans(ko)*iS*(x-p))*w);
    const matrix<double,3,3> H =
      - (2*wvn*iS*(x-p)*trans(w)
        - ( 2*wvn*wvn*iS - 2*wvn*wvn*iS*w*trans(w))
        - ( 2*wvn*w*trans(x-p)*iS
          - 6*double(trans(w)*iS*(x-p))*wvn*w*trans(w)
          - ( 2*wvn*wvn*w*(trans(w)*iS)
            - 2*wvn*wvn*double(trans(w)*iS*w)*w*trans(w)
            )
          + 2*wvn*double(trans(w)*iS*(x-p))*identity_matrix<double>(3)
          )
        );
    const matrix<double,3,1> wp = -inv(H)*dw;
    if (length(wp)<1e-10) return w+wp;
    size_t i;
    for (i=1;i!=1u<<8;i<<=1)
      if (wout_target(source,crystl,hkl,w+(wp/i))
         <wout_target(source,crystl,hkl,w)) break;
    //cerr << i << endl;
    //cerr << wout_target(source,crystl,hkl,w) << " :" << endl;
    //cerr << trans(w);
    //cerr << wout_target(source,crystl,hkl,w+(wp/i)) << " :" << endl;
    if (i>=(1u<<7)) return w;
    return optimize_wout(source,crystl,hkl,w+(wp/i));
  }

  matrix<double,3,1> optimize_wout(
      const source& source,
      const crystl& crystl,
      const IDX& hkl
      ){
    //cerr << "optimize_wout" << endl;
    const matrix<double,3,1> dhkl{
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)};
    const matrix<double,3,1> x = crystl.R*dhkl;
    return optimize_wout(
          source,
          crystl,
          hkl,
          normalize(x+source.kin)
          );
  }
  
  const double predict_integral(
      const struct source& source,
      const struct crystl& crystl,
      const IDX& hkl,
      const matrix<double,3,1> w
      ){
    const matrix<double,3,1> dhkl{
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)};
    const matrix<double,3,1> wi= normalize(source.kin);
    const matrix<double,3,1> x = crystl.R*dhkl;
    // normal to ewald sphere
    const matrix<double,3,1> p = length(source.kin)*w-source.kin;
    const matrix<double,3,3> S = source.Sw(w)+crystl.S(x);
    const matrix<double,3,3> psf =
      1e-8*(identity_matrix<double>(3)-w*trans(w));
    const matrix<double,3,1> z{0.0,0.0,1.0};
    const matrix<double,3,3> P0 =
      identity_matrix<double>(3)-w*trans(w);
    const matrix<double,3,2> P1 =
      matrix<double,3,2>{1.0,0.0,0.0,1.0,0.0,0.0};
    const matrix<double,2,2> iS2 =
      trans(P1)*(rotation_matrix(w,z)*(trans(P0)*inv(S)*P0))*P1;
    const double flx =
        get<0>(predict(w,zeros_matrix<double>(3,3),hkl,source,crystl))
      * 2*pi/sqrt(det(iS2));
    return flx;
  }

  const bool is_candidate(
      const struct source& source,
      const struct crystl& crystl,
      const IDX& hkl
      ){
    const matrix<double,3,1> dhkl{
      1.0*get<0>(hkl),
      1.0*get<1>(hkl),
      1.0*get<2>(hkl)};
    const matrix<double,3,1> x = crystl.R*dhkl;
    // normal to ewald sphere
    const matrix<double,3,1> w = normalize(x+source.kin);
    const matrix<double,3,1> p = length(source.kin)*w-source.kin;
    const double v =
        trans(w)*source.Sw(w)*w
      + trans(w)*crystl.S (x)*w
      + trans(w)*(crystl.R*trans(crystl.R))*w;
    const double d2 = length_squared(x-p);
    //cerr << d2 << " " << v << endl;
    return ((d2/v)<=1.0);
  }

  vector<tuple<int,int,int>> get_candidates
    (
      const struct source& source,
      const struct crystl& crystl
    )
    {
      vector<tuple<int,int,int>> todo;
      todo.emplace_back(0,0,0);
      patchmap<tuple<int,int,int>,void> done;
      vector<tuple<int,int,int>> candidates;
      while (todo.size()){
        // cerr << todo.size() << endl;
        const IDX hkl = todo.back(); todo.pop_back();
        if (done.count(hkl)) continue;
        done.insert(hkl);
        if (!is_candidate(source,crystl,hkl)) continue;
        todo.push_back({get<0>(hkl)-1,get<1>(hkl)  ,get<2>(hkl)  });
        todo.push_back({get<0>(hkl)+1,get<1>(hkl)  ,get<2>(hkl)  });
        todo.push_back({get<0>(hkl)  ,get<1>(hkl)-1,get<2>(hkl)  });
        todo.push_back({get<0>(hkl)  ,get<1>(hkl)+1,get<2>(hkl)  });
        todo.push_back({get<0>(hkl)  ,get<1>(hkl)  ,get<2>(hkl)-1});
        todo.push_back({get<0>(hkl)  ,get<1>(hkl)  ,get<2>(hkl)+1});
        candidates.push_back(hkl);
        continue;
      }
      return candidates;
    }

  auto predict_integrals(
      const vector<struct source>& sources,
      const struct crystl& crystl,
      const double& min_flx = exp(-1)
      ){
    vector<tuple<tuple<int,int,int>,double,matrix<double,3,1>>> prediction;
    const auto candidates = get_candidates(average_source(sources),crystl);
    prediction.reserve(candidates.size());
    for (const auto& hkl : candidates) {
      double flx = 0;
      for (const auto& source : sources)
        flx+= predict_integral
          (
            source,
            crystl,
            hkl,
            optimize_wout(source,crystl,hkl)
          );
      //cerr << flx << endl;
      if (flx<min_flx) continue;
      const matrix<double,3,1> w =
        optimize_wout(average_source(sources),crystl,hkl);
      prediction.emplace_back(hkl,flx,w);
    }
    cerr << "number of predictions: " << prediction.size() << endl;
    return prediction;
  }

  auto filter_and_project
  (
    const struct source& source,
    const struct crystl& crystl,
    const geometry& geom,
    const vector<tuple<IDX,double,matrix<double,3,1>>>& prediction,
    const double& min_flx = exp(-1)
  )
  {
    vector<tuple<
      IDX,
      double,
      const panel*,
      matrix<double,2,1>,
      matrix<double,2,2>
    >> projection;
    projection.reserve(prediction.size());
    for (const auto& [hkl,flx,w] : prediction)
    {
      const matrix<double,3,1> dhkl
      {
        1.0*get<0>(hkl),
        1.0*get<1>(hkl),
        1.0*get<2>(hkl)
      };
      const matrix<double,3,1>  x   = crystl.R*dhkl;
      const matrix<double,3,3>  S   = crystl.S(x)+source.Sw(w);
      const matrix<double,3,3> iS   = inv(S);
      for (const auto& panel : geom.panels)
      {
        const matrix<double,2,1> fs{1.0,0.0};
        const matrix<double,2,1> ss{0.0,1.0};
        const matrix<double,3,1> n =
          normalize(cross_product(panel.D*fs,panel.D*ss));
        //cerr << pow(double(trans(w)*n),2u) << endl;
        if (pow(double(trans(w)*n),2u)<0.01) continue;        
        matrix<double,3,1> y = w;
        matrix<double,2,1> fsss = panel(y);
        const matrix<double,3,2> P =
            (panel.D-y*trans(y)*panel.D/length_squared(y))
          * length(source.kin)/length(y);
        const matrix<double,3,1> v0 = x+source.kin-w*length(source.kin);
        // 
        //                               # trans(v0-P*x)*iS*(v0-P*x)
        //   trans(v0)*iS*v0
        // - trans(x)*trans(P)*iS*v0
        // - trans(v0)*iS*P*x
        // + trans(x)*trans(P)*iS*P*x
        // =                             # trans(v1-x)*trans(P)*iS*P*(v1-x) + c
        //   trans(v1)*trans(P)*iS*P*v1 + c
        //  -trans(x)*trans(P)*iS*P*v1
        //  -trans(v1)*trans(P)*iS*P*x
        //  +trans(x)*trans(P)*iS*P*x
        // 
        //   trans(x)*trans(P)*iS*v0   =   trans(x)*trans(P)*iS*P*v1
        //          trans(P*x)*iS*v0   =          trans(P*x)*iS*P*v1    
        //            trans(P)*iS*v0   =            trans(P)*iS*P*v1
        //                          v1 = inv(trans(P)*iS*P)*trans(P)*iS*v0
        //
        const matrix<double,2,2> PiSP   = trans(P)*iS*P;
        const matrix<double,2,2> iPiSP  = inv(PiSP);
        const matrix<double,2,2> pS     = iPiSP+identity_matrix<double>(2)/2;
        const matrix<double,2,2> ipS    = inv(pS);
        fsss+=iPiSP*trans(P)*iS*v0; // should be small...
        // flx*exp(-0.5*trans(a*fs)*ipS*(a*fs))/sqrt(det(2*pi*pS))
        // =
        // min_flx;
        // <->
        // # intersection
        // a=sqrt((2*log(flx)-2*log(min_flx)-log(det(2*pi*pS)))/trans(fs*ipS*fs))
        // # projection, this is what we want
        // a=sqrt((2*log(flx)-2*log(min_flx)-log(det(2*pi*pS)))*trans(fs*pS*fs))
        //const double tmp =
        //    (2*log(flx)-2*log(min_flx)-log(det(2*pi*pS)))
        //  * (trans(fs)*pS*fs);
        //if (tmp<=0.0) continue;
        //cerr << double(trans(fs)*pS*fs) << " " << trans(ss)*pS*ss << endl;
        const double a = sqrt(25*trans(fs)*pS*fs);
        const double b = sqrt(25*trans(ss)*pS*ss);
        //cerr << fsss(0) - a << " " << fsss(0) << " " << fsss(0) + a << endl;
        //cerr << fsss(1) - b << " " << fsss(1) << " " << fsss(1) + b << endl;
        const matrix<double,2,1> fsss0{
          fsss(0) - a,
          fsss(1) - b
        };
        const matrix<double,2,1> fsss1{
          fsss(0) + a,
          fsss(1) + b
        };
        if ( (fsss1(0)<0)
           | (fsss1(1)<0)
           | (panel.nfs<fsss0(0))
           | (panel.nss<fsss0(1))
           ) continue;
        //cerr << trans(fsss);
        //cerr << pS << endl;
        projection.emplace_back(hkl,flx,&panel,fsss,iPiSP);
      }
    }
    return projection;
  }

  auto predict
  (
    const vector<struct source>& sources,
    const struct crystl& crystl,
    const struct geometry& geom,
    const double& min_flx = exp(-1),
    const size_t oversampling = 1
  )
  {
    constexpr double max_var = 5*5;
    using reflection = vector<tuple<
          size_t,
          double,
          double,
          double,
          struct crystl,
          struct crystl,
          struct crystl
        >>;
    const auto source = average_source(sources);
    const auto projections =
      filter_and_project
        (
          source,
          crystl,
          geom,
          predict_integrals(sources,crystl,min_flx)
        );
    cerr << "number of projections after filter: " << projections.size() << endl;
    vector<tuple<IDX,reflection>> prediction;
    size_t counter = 0;
    for (const auto& [hkl,flx,p,m0,_S] : projections) {
      double integral = 0;
      //cerr << counter++ << " ";
      const matrix<double,2,2>  S = _S+identity_matrix<double>(2)/2;
      const matrix<double,2,2> iS = inv(S);
      //const double ext_fs = 8;//3*sqrt(S(0,0));
      const double ext_fs = sqrt(max_var*S(0,0));
      //const double ext_ss = 8;//3*sqrt(S(1,1));
      const double ext_ss = sqrt(max_var*S(1,1));
      const size_t min_fs = clip(int64_t(floor(m0(0)-ext_fs)),0,p->nfs-1);
      const size_t max_fs = clip(int64_t( ceil(m0(0)+ext_fs)),0,p->nfs-1);
      const size_t min_ss = clip(int64_t(floor(m0(1)-ext_ss)),0,p->nss-1);
      const size_t max_ss = clip(int64_t( ceil(m0(1)+ext_ss)),0,p->nss-1); 
      for (size_t ss=min_ss;ss<=max_ss;++ss) {
        for (size_t fs=min_fs;fs<=max_fs;++fs) {
          const matrix<double,2,1> fsss{1.0*fs+0.5,1.0*ss+0.5};
          const double tflx =
              flx
            * exp(-0.5*trans(fsss-m0)*iS*(fsss-m0))
            / sqrt(det(2*pi*S));
          if (trans(fsss-m0)*iS*(fsss-m0)>max_var) continue;
          //if (tflx<min_flx) continue;
          do
          {
            if (prediction.size())
              if (get<0>(prediction.back())==hkl) continue;
            prediction.emplace_back(hkl,reflection{});
          }while(false);
          get<1>(prediction.back()).emplace_back(
              (*p)(fs,ss),0.0,0.0,0.0,
              (struct crystl){},(struct crystl){},(struct crystl){}
              );
          vector<tuple<
            double,
            double,
            double,
            struct crystl,
            struct crystl,
            struct crystl
          >> p_predictions;
          p_predictions.reserve(oversampling*oversampling*sources.size());
          auto& [i,flx,wvn,bnd,dflx,dwvn,dbnd]=get<1>(prediction.back()).back();
          for (size_t oss=0;oss!=oversampling;++oss) {
            for (size_t ofs=0;ofs!=oversampling;++ofs) {
              const double o = oversampling;
              const matrix<double,2,1> x{fs+(2*ofs+1)/(2*o),ss+(2*oss+1)/(2*o)};
              const matrix<double,3,1> y = (*p)(x);
              const double t2 = 1.0/length_squared(y);
              const double t = sqrt(t2);
              const matrix<double,3,1> w = y*t;
              const matrix<double,3,2> D =
                  length(source.kin)
                * t*((p->D)+y*(trans(y)*(p->D))*t2);
              const matrix<double,2,2> psf_det{0.5/o,0.0,0.0,0.5/o};
              const matrix<double,3,3> psf = (D*psf_det)*trans(D*psf_det); 
              for (const auto& source : sources) {
                p_predictions.push_back
                (
                  predict
                  (
                    w,
                    psf,
                    hkl,
                    source,
                    crystl
                  )
                );
              }
            }
          }
          for (const auto& [pflx,pwvn,pbnd,pdflx,pdwvn,pdbnd] : p_predictions) {
            auto& [i,flx,wvn,bnd,dflx,dwvn,dbnd]=
              get<1>(prediction.back()).back();
            flx  += pflx;
            wvn  += pflx*pwvn;
            bnd  += pflx*pbnd;
            dflx += pdflx;
            dwvn += pdflx*pwvn+pdwvn*pflx;
            dbnd += pdflx*pbnd+pdbnd*pflx;
          }
          {
            matrix<double,3,1> v0 =
              normalize((*p)(matrix<double,2,1>{fs+0.0,ss+0.0}));
            matrix<double,3,1> v1 =
              normalize((*p)(matrix<double,2,1>{fs+1.0,ss+0.0}));
            matrix<double,3,1> v2 =
              normalize((*p)(matrix<double,2,1>{fs+0.0,ss+1.0}));
            matrix<double,3,1> v3 =
              normalize((*p)(matrix<double,2,1>{fs+1.0,ss+1.0}));
            const double area = length(v1-v0)*length(v2-v0);
            integral += flx*area;
          }
          wvn  /= flx;
          dwvn -= dflx*(wvn/flx);
          for (const auto& [pflx,pwvn,pbnd,pdflx,pdwvn,pdbnd] : p_predictions) {
            auto& [i,flx,wvn,bnd,dflx,dwvn,dbnd]=
              get<1>(prediction.back()).back();
            bnd  += pflx*pow(pwvn-wvn,2u);
            dbnd += pdflx*pow(pwvn-wvn,2u)+(pdwvn-dwvn)*pflx*2*(pwvn-wvn);
          }
          bnd  /= flx;
          dbnd -= dflx*(bnd/flx);
        }
      }
      //cerr << flx << " " << integral << endl;
      if (prediction.size())
        if (get<0>(prediction.back())==hkl) 
          if (integral<min_flx)
            prediction.pop_back();
    }
    return prediction;
  }

#if 0
  // predict the integrated partiality of the index hkl unconditionally
  double const inline predict(
      
    const matrix<double,3,1>& x,
      const matrix<double,3,3>& S,
      const source            & source
      )
  {
    // project x to ewald sphere
    const matrix<double,3,1> w = normalize(x+source.kin)
    const matrix<double,3,1> p = w*length(source.kin)-source.kin;
    // locally approximate ewald sphere as planar with given width
    const double ewald_width   = source.width(w);
    const matrix<double,3,3> S1= S + pow(ewald_width,2) * w * trans(w);
    const matrix<double,3,1> d = x - p;
    // TODO
    return source.flx*exp(-0.5*trans(d)*inv(S1)*(d))
          /(sqrt(det(2*pi*S))*source.area()*ewald_width);
  }

  // predict the integrated partiality of the index hkl unconditionally
  template<size_t verbosity>
  double const inline predict(
      const IDX& hkl,
      const vector<source>& sources,
      const struct crystl& crystl
      )
  {
    const matrix<double,3,1> x = crystl.R*matrix<double,3,1>(
          {1.0*get<0>(hkl),1.0*get<1>(hkl),1.0*get<2>(hkl)});
    const matrix<double,3,3> S = crystl.S(w);
    return accumulate(begin(sources),end(sources),0.0,
        bind(predict,x,S,_1));
  }
  
  // test if any reflection in {[h-0.5,h+0.5],[k-0.5,k+0.5],[l-0.5,k+0.5]}
  // can potentially produce a flux larger than fraction
  template<size_t verbosity>
  bool const inline is_candidate(
      const IDX& hkl,
      const source& source,
      const crystl& crystl,
      const geometry::geometry& geom,
      const double fraction
      )
  {
    const matrix<double,3,1> x = crystl.R*matrix<double,3,1>(
          {1.0*get<0>(hkl),1.0*get<1>(hkl),1.0*get<2>(hkl)});
    const matrix<double,3,3> S = crystl.S(w) + crystl.R*trans(crystl.R);
    // project x to ewald sphere
    const matrix<double,3,1> w = normalize(x+source.kin)
    const matrix<double,3,1> p = w*length(source.kin)-source.kin;
    // locally approximate ewald sphere as planar with given width
    const double ewald_width2  = pow(source.width(w),2);
    const matrix<double,3,3> S1= S + ewald_width2 * w * trans(w);
    const matrix<double,3,1> d = x - p;
    const double e             = trans(d)*inv(S1)*d;
    // TODO use fraction as a cutoff instead
    return (e<1.0);
  }

  template<size_t s>
  void inline predict(
      pair<uint32_t,tuple<double,double,double,crystl,crystl,crystl>>>& pixel,
      const matrix<double,3,1>& x, // reciprocal peak μ
      const matrix<double,3,3>& S0, // covariance of x
      const crystl crystl,
      const source source,
      const geometry::panel& panel,
      const double intensity = 1.0
      )
  {
    const double o = s;
    const auto& [fs,ss] = panel(pixel.first);
    auto& value  = pixel.second;
    auto& [flx,wvn,bnd,dflx,dwvn,dbnd] = pixel.second;
    for (size_t oss=0;oss!=s;++oss) for(size_t ofs=0;ofs!=s;++ofs) {
      // y is midpoint of this pixel on panel
      const matrix<double,3,1> y = panel(matrix<double,2,1>{fs+0.5,ss+0.5});
      // w is the direction-vector pointing to y
      const matrix<double,3,1> w = normalize(y);
      // p is the projection of w and y onto the ewald sphere
      const matrix<double,3,1> p = w*length(source.kin)-source.kin;
      const matrix<double,3,2>& D = panel.D;
      // S12o is the square root of the correlation matrix of p to smooth out
      // peaks with a variance equal to 1/oversampling to mitigate sampling
      // issues
      // S12o is not yet multiplied by the wavenumber and not yet divided by
      // o * length(y), this is being done in the next step when computing So
      // (D+y*(trans(y)*D)/(length_squared(y)))/length_squared(y) is the
      // derivative of w wrt to fs,ss
      const matrix<double,3,2> S12o = D+y*(trans(y)*D)/length_squared(y);
      // So is the correalation matrix of p
      const matrix<double,3,3> So =
        length_squared(source.kin)/(o*o*length_sqared(y))*S12o*trans(S12o);
      // S1 is the total covariance of the source in reciprocal space in
      // direction w
      const matrix<double,3,3> S1 = source.Sw(w) + So;
      // coviariance matrices describing normalisation (S_circ) and 
      // shape (S_star) of product
      const matrix<double,3,3> S_circ = S0+S1;
      const matrix<double,3,3> S0i = inv(S0i);
      const matrix<double,3,3> S1i = inv(S1i);
      // S_stari is the inverse of S_star
      const matrix<double,3,3> S_stari= S0i+S1i;
                       // b-factor
      const double _flx = crystl.scaling(x)
                       // structure factor
                         *intensity
                       // exponential
                         *exp(-0.5*trans(x-p)*S_circ*(x-p))
                       // normalisation
                         *(sqrt(det(S_stari/(2*pi)))/source.area());
      struct crystl dflx;
      dflx.a = crystl.scaling_da(x)
              *intensity
              *exp(-0.5*trans(x-p)*S_circ*(x-p))
              *(sqrt(det(S_stari/(2*pi)))/source.area());
      dflx.b = crystl.scaling_db(x)
              *intensity
              *exp(-0.5*trans(x-y)*S_circ*(x-p))
              *(sqrt(det(S_stari/(2*pi)))/source.area());
      dflx.U = ;
      const matrix<double,3,1> m_star = S_star * ( S0i*x + S1i*p );
      // the expected wavelength is the length of m_star measured from -kin
      const double _wvn = length(m_star + source.kin);
      // the bandwidth is the width of the exponential projected in the direction
      // of the outgoing wave vector w (normed)
      const double _bnd = trans(w)*S_circ*w;
    }
  }
  
  // predict the reflection of the index hkl unconditionally
  template<size_t oversampling>
  //           pixel    flx    wvn    bnd    dflx   dwvn   dbnd
  void inline predict(
      const matrix<double,3,1>& x,
      const source& source,
      const struct crystl& crystl,
      const geometry::geometry& geom,
      patchmap& shape,
      const double intensity = 1.0
      )
  {
    if (shape.size()==0) {
    } else {
      for 
    }
    return accumulate();
  }
  
  // predict the reflection of the index hkl unconditionally
  template<size_t oversampling>
  //           pixel    flx    wvn    bnd    dflx   dwvn   dbnd
  auto const inline predict(
      const IDX hkl,
      const vector<source>& sources,
      const struct crystl& crystl,
      const geometry::geometry& geom,
      const double intensity = 1.0
      )
  {
    patchmap<uint32_t,tuple<double,double,double,crystl,crystl,crystl>> shape;
    accumulate(sources.begin(),sources.end(),shape,predict<oversampling>);
    return vector(shape.begin(),shape.end);
  }

  // predict the reflection of the index hkl unconditionally
  template<size_t verbosity,size_t oversampling>
  //           pixel    flx    wvn    bnd    dflx   dwvn   dbnd
  vector<tuple<uint32_t,double,double,double,crystl,crystl,crystl>>{};
  const inline predict(
      const IDX hkl,
      const vector<source>& sources,
      const struct crystl& crystl,
      const geometry::geometry& geom,
      const double intensity = 1.0
      )
  {
    vector<tuple<uint32_t,double,double,double,crystl,crystl,crystl>> prediction;
    // TODO: predict reflection
    return prediction;
  }
  
  template<predict_mode mode=predict_mode::pixel_partiality,
           bool compute_derivatives=true,
           size_t verbosity=0,
           size_t  oversampling=2>
  auto const inline predict( // one template to rule them all
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = exp(-1)              // minimum flux
      )
  {
    vector
    <
      invoke_result
      <
        get_predict_return_type<mode,compute_derivatives>>::type
      >
    > prediction;
    vector<IDX> todo;
    unordered_set<IDX> done;
    transform(
         geom.panels.begin(), 
         geom.panels.end(),
         back_inserter(todo),
         seed_todo
        );
    while (todo.size()) {
      if (done.count(todo.back())) continue;
      const IDX hkl = todo.back(); todo.pop_back();
      const auto [ is_candidate, prediction_hkl ] =
        predict<predict_mode,compute_derivatives,verbosity,oversampling>(
            hkl,
            sources,
            crystl,
            geom,
            fraction);
      if (!is_candidate) continue;
      prediction.insert(prediction_hkl);
      todo.push_back({get<0>(hkl)-1,get<1>(hkl)  ,get<2>(hkl)  });
      todo.push_back({get<0>(hkl)+1,get<1>(hkl)  ,get<2>(hkl)  });
      todo.push_back({get<0>(hkl)  ,get<1>(hkl)-1,get<2>(hkl)  });
      todo.push_back({get<0>(hkl)  ,get<1>(hkl)+1,get<2>(hkl)  });
      todo.push_back({get<0>(hkl)  ,get<1>(hkl)  ,get<2>(hkl)-1});
      todo.push_back({get<0>(hkl)  ,get<1>(hkl)  ,get<2>(hkl)+1});
    }
    return prediction;
  }

  template<predict_mode mode=predict_mode::candidate_hkls,
           bool compute_derivatives=false,
           size_t verbosity=0,
           size_t  oversampling=2>
  auto const inline predict( // one template to rule them all
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = exp(-1)              // minimum flux
      ){
    if constexpr (verbosity>3){
      cerr << "calling predict" << endl;
    }
    if constexpr (verbosity>3){
      if constexpr (mode==predict_mode::candidate_hkls){
        cerr << "candidate_hkls" << endl;
      }
      if constexpr (mode==predict_mode::pixel_partiality){
        cerr << "pixel_partiality" << endl;
      }
      if constexpr (mode==predict_mode::index_partiality){
        cerr << "index_partiality" << endl;
      }
      if constexpr (compute_derivatives){
        cerr << "with derivatives" << endl;
      } else {
        cerr << "without derivatives" << endl;
      }
      cerr << "verbosity = " << verbosity << endl;
      cerr << "oversampling = " << oversampling << endl;
    }
    const double max_e = -sqrt(2)*inverf(fraction);
    const double max_e2= pow(max_e,2);
    const auto source = average_source(sources);
    if constexpr (verbosity>3){
      cerr << "source.win =" << endl;
      cerr << source.win << endl;
      cerr << "crystl.R =" << endl;
      cerr << crystl.R << endl;
      cerr << "min_flux = " << min_flux << endl;
    }
    const matrix<double,3,1> kin = source.win*source.wvn;
    if constexpr (verbosity>3){
      cerr << "crystl.R = " << endl << crystl.R << endl;
      cerr << "max_e = " << max_e << endl;
      cerr << "kin = " << endl << kin << endl;
    }
    unordered_set<IDX,hash_functor<IDX>> hkls;
    auto prediction = get_predict_return_type<mode,compute_derivatives>();
    const double n0 = sqrt(2*pi*det(crystl.peak));
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
    vector<IDX> todo;
    unordered_set<IDX,hash_functor<IDX>> done;
    // take one point on a panel
    const matrix<double,2,1> x{it->nfs/2.0,it->nss/2.0};
    // project to detector
    const matrix<double,3,1> y   = (*it)(x);
    if constexpr (verbosity>3) cerr << it->D << endl;
    if constexpr (verbosity>3) cerr << it->o << endl;
    const matrix<double,3,1> Dw  = normalize(y)-source.win;
    const matrix<double,3,1> hkl = crystl.U*Dw*source.wvn;
    // find closest hkl value
    const int32_t h = round(hkl(0)), k = round(hkl(1)), l= round(hkl(2));
    if constexpr (verbosity>3){
      cerr << "seeding todo with:" << endl;
      cerr << h << " " << k << " " << l << endl;
    }
    const IDX first_index({h,k,l});
    todo.push_back({h,k,l});
    while (todo.size()){
    const IDX index = todo.back();
    todo.pop_back();
    if (verbosity>3) cerr << "todo.pop_back()" << endl;
    for (int32_t l=get<2>(index)-1;l<=get<2>(index)+1;++l)
    for (int32_t k=get<1>(index)-1;k<=get<1>(index)+1;++k)
    for (int32_t h=get<0>(index)-1;h<=get<0>(index)+1;++h){
      if constexpr (verbosity>3)
        cerr  << "todo.size() = " << todo.size() << endl;
      if constexpr (verbosity>3)
        cerr << "testing " << h << " " << k << " " << l << endl;
      const IDX index{h,k,l};
      if (done.count(index)){
        if constexpr (verbosity>3)
          cerr << "index already visited" << endl;
        continue;
      }
      done.insert(index);
      const matrix<double,3,1> hkl{static_cast<double>(h),
                                   static_cast<double>(k),
                                   static_cast<double>(l)};
      const matrix<double,3,1> m   = crystl.R*hkl;
            matrix<double,3,1> wout= normalize(m+kin);
            matrix<double,3,1> Dw  = wout-source.win;
            matrix<double,3,1> Dk  = source.wvn*Dw;
      const matrix<double,3,3> S1  =
        max_e*covariance_matrix(Dw,m,source,crystl)+SR;
      const double e1= trans(m-Dk)*inv(S1)*(m-Dk);
      if ( e1 > 1 ){
        if constexpr (verbosity==3) // rejected because of e1
          cerr << h << " " << k << " " << l << " 1 " << e1 << endl;
        continue;
      }
      matrix<double,3,1> y = wout;
      matrix<double,2,1> fsss = (*it)(y);
      bool wasvalid = it->isvalid(fsss); 
      for (size_t j=0;j!=4;++j){
        bool success = true;
        wout = optimize_wout<verbosity>(wout,m,source,crystl,success);
        if (!success) break;
      }
      fsss = (*it)(y=wout);
      if (!it->isvalid(fsss)){
        if constexpr (verbosity==3) // peak will not be on panel
          cerr << h << " " << k << " " << l << " 2 " << " "
               << fsss(0) << " " << fsss(1) << endl;
        if constexpr (verbosity==2)
          if (index==first_index)
            cerr << "peak will not be on panel " << endl;
        if (wasvalid) todo.push_back(index);
        continue;
      } else if constexpr (verbosity>3){
        cerr << y(0) << " " << y(1) << " " << y(2) << endl;
        cerr << fsss(0) << " " << fsss(1) << endl;
      }
      todo.push_back(index);
      Dw   = wout-source.win;
      Dk   = source.wvn*Dw;
      const matrix<double,3,3>  S = covariance_matrix(Dw,m,source,crystl);
      //cerr << "S=" << endl;
      //cerr << S << endl;
      const matrix<double,3,3> iS = inv(S);
      //cerr << "iS=" << endl;
      //cerr << iS << endl;
      const double e0 = trans(Dk-m)*iS*(Dk-m);
      if constexpr (verbosity>3){
        //cerr << "length(hkl-x) = " << length(hkl-x) << endl;
        cerr << S << endl;
        cerr << S1 << endl;
        cerr << m << endl;
        cerr << Dk << endl;
        cerr << "e1 = " << e1 << endl
             << "e0 = " << e0 << endl;
      }
      if constexpr (verbosity==2){
        if (index==first_index){
          cerr << "testing " << h << " " << k << " " << l << endl;
          cerr << S << endl;
          cerr << S1 << endl;
          cerr << m << endl;
          cerr << Dk << endl;
          cerr << "e1 = " << e1 << endl
               << "e0 = " << e0 << endl;
        }
      }
      if constexpr (verbosity>3)
        cerr << "queuing hkl for test " << h << " " << k << " " << l << endl;
      if ( e0  > 2*max_e ){
        if constexpr (verbosity==3)
          cerr << h << " " << k << " " << l << " 3 " << e0 << endl;
        continue;
      }
      if constexpr (verbosity>3)
        cerr << "accepted " << h << " " << k << " " << l << " "
             << y(0) << " " << y(1) << " " << y(2)<< endl;
      hkls.insert(index);
      if constexpr(mode==predict_mode::candidate_hkls) {continue;} else{
      matrix<double,3,1>* wouts =
        new matrix<double,3,1>[sources.size()]();
      double partiality = 0;
      double normalization = 0; // normalizing the partiality to a sane value
      if constexpr (mode==predict_mode::index_partiality)
        prediction.push_back({index,0,0,0});
      for (size_t j=0;j!=1+2*(mode==predict_mode::index_partiality);++j)
      for (size_t i=0;i!=sources.size();++i){
        const auto& source = get<1>(sources[i]);
        const matrix<double,3,1> kin = source.win*source.wvn;
        auto& wout = wouts[i];
        if (j==0){
          wout = normalize(m+kin);
          for (size_t j=0;j!=6;++j){
            bool success = true;
            const matrix<double,3,1> tmp =
            optimize_wout(wout,m,source,crystl,success);
            if (!success) break;
            wout = tmp;
          }
        }
        const matrix<double,3,1> Dw = wout-source.win;
        const matrix<double,3,1> Dk = source.wvn*Dw;
        const matrix<double,3,3> SS = matrix_S(Dw,source);
        const matrix<double,3,3> iSS= inv(SS);
        const matrix<double,3,3> SP = matrix_P(m,crystl);
        const matrix<double,3,3> iSP= inv(SP);
        const matrix<double,3,3> S0 = SS+SP;
        const matrix<double,3,3> iS0= inv(S0);
        const matrix<double,3,3> iS1 = iSS+iSP;
        const matrix<double,3,3> S1  = inv(iSS+iSP);
        const double e = 0.5*trans(m-Dk)*iS0*(m-Dk);
        //const double e = 0.5*(log(abs(det(SP)/det(SS)))
        //                     -3
        //                     +trace(iSP*SS)
        //                     +trans(m-Dk)*iSP*(m-Dk));
        //cout << "e= " << e << endl;
        if (e>=LOG_DBL_MAX) continue;
        const double v = trans(wout)*S0*wout;
        const double p = exp(-e)/sqrt(2*pi*v);
        if (j==0){
          partiality    += get<0>(sources[i])*p;
          normalization += get<0>(sources[i])/det(SS+SR);
        } else if constexpr (mode==predict_mode::index_partiality){
          const matrix<double,3,1> mu = S1*(iSP*m+iSS*Dk);
          const double w = get<0>(sources[i])*p/partiality;
          const double lDw = length(Dw);
          const double ewvn = length(mu)/lDw;
          if constexpr (verbosity>3){
            cerr << "inside second loop, about to calculate"
                 << "expected wavenumber" << endl;
            cerr << w << " " << lDw << " " << length(mu) << " "
                 << source.wvn << endl;
            cerr << mu << endl;
          }
          if (j==1){
            get<2>(prediction.back())+=w*ewvn;
          }
          if (j==2){
            const matrix<double,3,1> nDw = Dw/lDw;
            get<3>(prediction.back())+=
              w*(pow(ewvn-get<2>(prediction.back()),2)
                +1.0/(trans(nDw)*iS1*nDw));
            if constexpr (verbosity>3){
              cerr << "bandwidth of reflection on detector" << endl;
              cerr << sqrt(pow(ewvn-get<2>(prediction.back()),2)) << " "
                   << sqrt(1.0/(trans(nDw)*iS1*nDw)) << endl;
            }
          }
        } else continue;
      }
      partiality = partiality/(partiality+normalization);
      //cerr << partiality << " " << fraction << endl;
      if (partiality<1e-9) {
        if constexpr (verbosity==3)
          cerr << h << " " << k << " " << l << " 4 "
               << partiality << endl;
        if constexpr (mode==predict_mode::index_partiality)
          prediction.pop_back();
        delete[] wouts;
        continue;
      }
      //cerr << h << " " << k << " " << l << " 0 " << partiality << endl;
      if constexpr (verbosity==3)
        cerr << h << " " << k << " " << l << " 0 "
             << partiality/(partiality+normalization) << endl;
      if constexpr(mode==predict_mode::index_partiality){
      get<1>(prediction.back())=partiality/(partiality+normalization);
      get<4>(prediction.back())=fsss(0);
      get<5>(prediction.back())=fsss(1);
      delete[] wouts;
      continue;
      } else if constexpr(mode==predict_mode::pixel_partiality) {
      const double o = oversampling;
      const double dpsfv = pow(o,-2); // +1;
      unordered_map<
        uint32_t,
        typename conditional<
          compute_derivatives,
          tuple<double,double,double,struct crystl,struct crystl,struct crystl>,
          tuple<double,double,double> // sumw mean M2
        >::type
      > shape;
      unordered_map<size_t,array<double,2>> helper;
      matrix<double,3,1>  y      = wout;
      matrix<double,2,1>  fsss   = (*it)(y);
      const double norm_y        = length(y);
      const matrix<double,3,2> P =
        it->D/norm_y-y*trans(y)*it->D/pow(norm_y,3u);
      // TODO fix m0
      //const matrix<double,3,1>   m0 = source.wvn*(P*x+source.win-y/norm_y)+m;
      //cerr << " m0 and wvn*wout should roughly be the same " << endl;
      //cerr << m0 << endl;
      //cerr << source.wvn*wout << endl; 
      //cerr << "iS=" << endl;
      //cerr << iS << endl;
            matrix<double,2,2>  iAS1 = trans(P)*iS*P;
      const matrix<double,2,2>   AS1 =
        inv(iAS1)+dpsfv*identity_matrix<double>(2);
                                iAS1 = inv(AS1);
      //cerr << "AS1=" << endl;
      //cerr << AS1 << endl;
      const matrix<double,2,2>  _AS1 = max_e2*AS1; 
      const matrix<double,2,2> _iAS1 = inv(_AS1);
      //cerr << "_AS1=" << endl;
      //cerr << _AS1 << endl;
      double * partialities;
      double * ewvns;
      if constexpr (compute_derivatives){
        partialities = new double[sources.size()];
        ewvns        = new double[sources.size()];
      }
      for (size_t j=0;j!=1+compute_derivatives;++j)
      for (size_t i=0;i!=sources.size();++i){
        const auto& source = get<1>(sources[i]);
        const matrix<double,3,1>  kin = source.win*source.wvn;
        const matrix<double,3,3>    S = covariance_matrix(Dw,m,source,crystl);
 /* map k to the detector whos coordinate system is given by:                *
  *                                                                          *
  *  / fs2x  ss2x  \  / fs \   / corner_x \  =  / x \                        *
  *  | fs2y  ss2y  |  \ ss / + | corner_y |  =  | y |                        *
  *  \ fs2z  ss2z  /           \ corner_z /  =  \ z /                        *
  *                                                                          *
  *   *  matrix D *   * x *     * offset *      * y *                        */
        matrix<double,3,1>          y = wouts[i];
        matrix<double,2,1>       fsss = (*it)(y);
        const double           norm_y = length(y);
        const matrix<double,3,2>    P =
          it->D/norm_y-y*trans(y)*it->D/pow(norm_y,3u);
      // TODO fix m0
      //const matrix<double,3,1>   m0 = source.wvn*(P*x+source.win-y/norm_y)+m;
        const matrix<double,2,1>   m0 = fsss;
        //cerr << "m0=" << endl;
        //cerr << m0 << endl;
        const matrix<double,2,2>  iS1 = trans(P)*iS*P;
        const matrix<double,2,2>   S1 = inv(iS1);
        //cerr << "S1 =" << endl; 
        //cerr << S1 << endl; 
        const matrix<double,2,2>  _S1 =
          max_e2*(S1+dpsfv*identity_matrix<double>(2));
        const matrix<double,2,2> _iS1 = inv(_S1);
        const double v = trans(wout)*S*wout;
        const double ext_fs = sqrt(min(_S1(0,0),_AS1(0,0)));
        const double ext_ss = sqrt(min(_S1(1,1),_AS1(1,1)));
        const size_t min_fs = clip(int64_t(floor(m0(0)-ext_fs)),0,it->nfs-1);
        const size_t max_fs = clip(int64_t( ceil(m0(0)+ext_fs)),0,it->nfs-1);
        const size_t min_ss = clip(int64_t(floor(m0(1)-ext_ss)),0,it->nss-1);
        const size_t max_ss = clip(int64_t( ceil(m0(1)+ext_ss)),0,it->nss-1);
        //cerr << min_fs << " " << max_fs << " "
        //     << min_ss << " " << max_ss << endl;
        for (size_t ss=min_ss;ss<=max_ss;++ss)
            for (size_t fs=min_fs;fs<=max_fs;++fs){
        const matrix<double,2,1> x{fs+0.5,ss+0.5};
        //cerr << "x =" << endl; 
        //cerr << x << endl; 
        //cerr << trans(x-m0)*_iAS1*(x-m0) << endl;
        //if (trans(x-m0)*_iAS1*(x-m0)+e0>1) continue;
        const size_t p = (*it)(fs,ss);
        if (trans(x-m0)*iAS1*(x-m0)+e0>max_e2){
          //shape[p]={0,length(m+source.win*source.wvn),0};
          continue;
        }
        if (trans(x-m0)*_iS1*(x-m0)>1) continue;
        for (size_t oss=0;oss!=oversampling;++oss)
            for(size_t ofs=0;ofs!=oversampling;++ofs){
        const matrix<double,2,1> x{fs+(2*ofs+1)/(2*o),ss+(2*oss+1)/(2*o)};
        const matrix<double,3,1> y = (*it)(x);
        const double        norm_y = length(y);
        const matrix<double,3,2> P =
          it->D/norm_y-y*trans(y)*it->D/pow(norm_y,3u);
        const matrix<double,3,1> wout = y/norm_y;
        const matrix<double,3,3>    I =
          dpsfv*pow(source.wvn,2)*P*trans(P); 
        const matrix<double,3,1>   Dw = wout-source.win;
        const matrix<double,3,1>    k = source.wvn*wout;
        const matrix<double,3,1>   Dk = source.wvn*Dw;
        const matrix<double,3,3>   SS = matrix_S(Dw,source);
        const matrix<double,3,3>  iSS = inv(SS);
        const matrix<double,3,3>   SP = matrix_P(m,crystl);
        const matrix<double,3,3>  iSP = inv(SP);
        const matrix<double,3,3>   S0 = SS+SP+I ;
        const matrix<double,3,3>  iS0 = inv(S0);
        const matrix<double,3,3>  iS2 = iSS+iSP;
        const matrix<double,3,3>   S2 = inv(iSS+iSP);
        const matrix<double,3,1> diff = m-Dk;
        const double e = 0.5*trans(diff)*iS0*diff;
      //const double               e2 = 0.5*trans(x-m0)*iS1*(x-m0);
        /*cerr << "Dk = " << endl;
        cerr << Dk << endl;
        cerr << "m = " << endl;
        cerr << m << endl;
        cerr << "I=" << endl;
        cerr << I << endl;
        cerr << "S0=" << endl;
        cerr << S0 << endl;*/
        //cerr << "e= " <<  e  << " e2= " << e2 << endl;
        //if (e>max_e2) continue;
        if (e>LOG_DBL_MAX){
          //shape[p]={0,length(m+source.win*source.wvn),0};
          continue;
        }
        //if (shape.count(p)==0) shape[p]={0,length(m)/length(Dw),0};
        const double v = trans(wout)*S0*wout;
        const matrix<double,3,1> mu = S2*(iSP*m+iSS*Dk);
        // x = iSP*(y + SP*iSS*z)
        //const double normalization = 1.0/det(SS+SR);
        const double partiality =
          get<0>(sources[i])*exp(-e)
        // *sqrt(det(SS+SR)/(pow(2*pi,3)*v)); 
        // The normalization is arbitrary to some extent,
        // it shall be normalized in 2 dimensions therefore normalization
        // is 2pi not (2pi)¹·⁵
          *sqrt(det(SS+SR)/(pow(2*pi,2)*v));
        //if (partiality<1e-10) continue;
        //cerr << partiality << endl;
        const double lDw  = length(Dw);
        //const double ewvn = length(mu)/lDw;
        const matrix<double,3,1> nDw = Dw/lDw;
        const double v0   = (trans(nDw)*SP*nDw);
        const double v1   = (trans(nDw)*SS*nDw);
        const double w0   = 1.0/v0;
        const double w1   = 1.0/v1;
        const double ewvn = (w0*length(m)+w1*length(Dk))/((w0+w1)*lDw);
        const double bnd  = 1.0/(trans(nDw)*iS2*nDw);
        size_t n;
        if constexpr (compute_derivatives){
          n = i;
          n*= it->nss*oversampling;
          n+= ss*oversampling+oss;
          n*=    it->nfs*oversampling;
          n+= fs*oversampling+ofs;
        }
        if (j==0){
          mean_variance(
              ewvn,             // x
              partiality,       // w
              get<0>(shape[p]), // sumw
              get<1>(shape[p]), // mean
              get<2>(shape[p])  // M2
              );
          get<2>(shape[p]) += bnd*partiality;
          if constexpr (compute_derivatives) {
            get<0>(helper[n]) = ewvn;
            get<1>(helper[n]) = partiality;
          }
        } else {
          // const double v = trans(wout)*(SS+SP+I)*wout;
          // const double partiality = get<0>(sources[i])*
          // exp(-0.5*trans(R*hkl-Dk)*inv(SS+SP+I)*(R*hkl-Dk))
          // *sqrt(det(SS+trans(R)*R)/(pow(2*pi,3)*trans(wout)*(SS+SP+I)*wout));
          // derivative of partiality wrt. R = 
          // -partiality*
          const double m2 = pow(crystl.mosaicity,2);
          const double s2 = pow(crystl.strain,2);
          const double detSSpSR = det(SS+SR);
          const double t0 = sqrt(detSSpSR/(2*pi*pow(v,3)));
          const matrix<double,3,3> wouthkl = wout*trans(hkl);
          const double             twm     = trans(wout)*m;
          const matrix<double,3,1> iS0diff = iS0*diff;
          const matrix<double,3,3> partiality_dR =
            -(iS0diff
             -m2*trans(iS0diff)*iS0diff*m
             +m2*trans(iS0diff)*m*iS0diff
             -s2*trans(iS0diff)*m*iS0diff
             )*trans(hkl)*partiality
            +get<0>(sources[i])*exp(-e)*
             (detSSpSR*inv(SS+SR)*crystl.R/sqrt(detSSpSR*2*pi*v)
             -m2*sqrt(detSSpSR/(2*pi*pow(v,3)))*
              (m*trans(hkl)+twm*wout*trans(hkl))
             -2*pi*s2*twm*t0*wouthkl
             );
          const double nx2 = pow(length(crystl.R*hkl),2);
          const double partiality_dmosaicity =
            m*partiality*trans(diff)*iS0
            *(nx2*identity_matrix<double>(3)-m*m)
            *iS0*diff
            -get<0>(sources[i])*exp(-e)
              *2*crystl.mosaicity*t0*trans(wout)
              *(nx2*identity_matrix<double>(3)-m*m)*wout;
          const double partiality_dstrain =
            crystl.strain*partiality*trans(hkl)
              *trans(crystl.R)*iS0*diff*trans(diff)*iS0*m
            -get<0>(sources[i])*exp(-e)
              *2*crystl.strain*t0*trans(hkl)*trans(crystl.R)*m;
          //const double partiality_d
          const double mean_dw = mean_variance_mean_dw(
              ewvn,
              partiality,
              get<0>(shape[p]), // sumw
              get<1>(shape[p]), // mean
              get<2>(shape[p])  // M2
              );
          const double mean_dx = mean_variance_mean_dx(
              ewvn,
              partiality,
              get<0>(shape[p]), // sumw
              get<1>(shape[p]), // mean
              get<2>(shape[p])  // M2
              );
          const double M2_dw_ini = mean_variance_M2_dw_ini(
              ewvn,
              partiality,
              get<0>(shape[p]), // sumw
              get<1>(shape[p]), // mean
              get<2>(shape[p])  // M2
              );
          const double M2_dx_ini = mean_variance_M2_dx_ini(
              ewvn,
              partiality,
              get<0>(shape[p]), // sumw
              get<1>(shape[p]), // mean
              get<2>(shape[p])  // M2
              );
        }
        //get<0>(shape[p])=1;
        }
        }
      }
      vector<
        typename conditional<
          compute_derivatives,
          tuple<uint32_t,
                double,struct crystl,
                double,struct crystl,
                double,struct crystl>,
          tuple<uint32_t,double,double,double>
        >::type
      > unrolledshape;
      for (auto it=shape.begin();it!=shape.end();++it){
        if constexpr (compute_derivatives){
          unrolledshape.push_back(
              {
              it->first,
              get<0>(it->second),
              get<1>(it->second),
              get<2>(it->second),
              get<3>(it->second),
              get<4>(it->second),
              get<5>(it->second)
              });
        } else {
          //if (get<0>(it->second)>1e-8)
          unrolledshape.push_back(
              {
              it->first,
              get<0>(it->second),
              get<1>(it->second),
              get<2>(it->second)
              });
        }
        }
        prediction.push_back(std::make_tuple(index,unrolledshape));
      }
      delete[] wouts;
      }
    }
    }
    }
    if constexpr (mode==predict_mode::candidate_hkls) return hkls;
    else return prediction;
  }

  template<size_t verbosity=0>
  auto const inline candidate_hkls(
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = 1.0/64               // minimum partiality
      ){
    return predict<predict_mode::candidate_hkls,false,verbosity,0>
      (sources,crystl,geom,fraction);
  }
  
  template<size_t verbosity=0>
  auto const inline index_partiality(
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = 1.0/64               // minimum partiality
      ){
    return predict<predict_mode::index_partiality,false,verbosity,0>
      (sources,crystl,geom,fraction);
  }
  
  template<bool compute_derivatives = false>
  auto const inline pixel_partiality(
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = 1.0/16               // minimum partiality
      ){
    return predict<predict_mode::pixel_partiality,compute_derivatives,0,1>
      (sources,crystl,geom,fraction);
  }

  template<class ifstream>
  vector<tuple<double,source>> const inline deserialize_sources(ifstream& file){
    vector<tuple<double,source>> sources;
    uint64_t n;
    file.read(reinterpret_cast<char*>(&n),8);
    sources.resize(n); 
    for (size_t i=0;i!=n;++i){
      struct source& source = get<1>(sources[i]);
      file.read(reinterpret_cast<char*>(&get<0>(sources[i]   )),8);
      file.read(reinterpret_cast<char*>(&source.win(0)        ),8);
      file.read(reinterpret_cast<char*>(&source.win(1)        ),8);
      file.read(reinterpret_cast<char*>(&source.win(2)        ),8);
      file.read(reinterpret_cast<char*>(&source.wvn           ),8);
      file.read(reinterpret_cast<char*>(&source.bnd           ),8);
      double div;
      file.read(reinterpret_cast<char*>(&div),8);
      source.div = pow(div,2u)*(identity_matrix<double>(3)
                   -source.win*trans(source.win));
    }
    return sources;
  }

  template<class ifstream>
  vector<crystl> const inline deserialize_crystls(ifstream& file){
    vector<crystl> crystls;
    uint64_t n;
    file.read(reinterpret_cast<char*>(&n),8);
    crystls.resize(n); 
    for (size_t i=0;i!=n;++i){
      struct crystl& crystl = crystls[i];
      double peak;
      file.read(reinterpret_cast<char*>(&crystl.R(0,0)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(0,1)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(0,2)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(1,0)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(1,1)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(1,2)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(2,0)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(2,1)   ),8);
      file.read(reinterpret_cast<char*>(&crystl.R(2,2)   ),8);
      crystl.U = inv(crystl.R);
      file.read(reinterpret_cast<char*>(&crystl.mosaicity),8);
      file.read(reinterpret_cast<char*>(&peak            ),8);
      file.read(reinterpret_cast<char*>(&crystl.strain   ),8);
      crystl.peak = pow(peak,2u)*identity_matrix<double>(3);
      file.read(reinterpret_cast<char*>(&crystl.a        ),8);
      file.read(reinterpret_cast<char*>(&crystl.b        ),8);
    }
    return crystls;
  }
 
  template<class ifstream,class ofstream>
  const inline void sources_ascii2bin(ifstream& in,ofstream& out){
    vector<array<double,7>> buffer;
    while (in){
      if (in.peek()=='>'){
        in.get();
        buffer.push_back({});
        for (size_t i=0;i!=7;++i){
          in >> buffer.back()[i];
        }
        in.ignore(numeric_limits<std::streamsize>::max(),'\n');
      } else {
        break;
      }
    }
    const uint64_t n = buffer.size();
    out.write(reinterpret_cast<const char*>(&n),8);
    for (auto it=buffer.begin();it!=buffer.end();++it)
      for (size_t i=0;i!=7;++i)
        out.write(reinterpret_cast<char*>(&((*it)[i])),8);
  }
  
  template<class ifstream,class ofstream>
  const inline void crystls_ascii2bin(ifstream& in,ofstream& out){
    vector<array<double,14>> buffer;
    while (in){
      if (in.peek()=='<'){
        in.get();
        buffer.push_back({});
        for (size_t i=0;i!=14;++i) in >> buffer.back()[i];
        in.ignore(numeric_limits<std::streamsize>::max(),'\n');
      } else {
        break;
      }
    }
    const uint64_t n = buffer.size();
    out.write(reinterpret_cast<const char*>(&n),8);
    for (auto it=buffer.begin();it!=buffer.end();++it)
      for (size_t i=0;i!=14;++i)
        out.write(reinterpret_cast<char*>(&((*it)[i])),8);
  }
#endif
}
#endif // PARTIALITY_H
