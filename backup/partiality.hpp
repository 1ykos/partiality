#include "wmath.hpp"
#include "encode.hpp"
#include "geometry.hpp"

#include "dlib/matrix.h"
#include "dlib/optimization.h"

namespace partiality{
constexpr double pi          = 3.14159265358979323846;
constexpr double LOG_DBL_MAX = 709.78271289338402993962517939506;
//using dlib::sqrt;
using dlib::abs;
using dlib::cholesky_decomposition;
using dlib::derivative;
using dlib::diag;
using dlib::diagm;
using dlib::dot;
using dlib::eigenvalue_decomposition;
using dlib::identity_matrix;
using dlib::inv;
using dlib::length;
using dlib::length_squared;
using dlib::make_symmetric;
using dlib::matrix;
using dlib::matrix_exp;
using dlib::matrix_op;
using dlib::normalize;
using dlib::op_make_symmetric;
using dlib::round;
using dlib::set_colm;
using dlib::tmp;
using dlib::trace;
using dlib::zeros_matrix;
using std::abs;
using std::array;
using std::bitset;
using std::cerr;
using std::cin;
using std::complex;
using std::cout;
using std::endl;
using std::fill;
using std::fixed;
using std::floor;
using std::get;
using std::getline;
using std::ifstream;
using std::isnan;
using std::istream;
using std::map;
using std::max_element;
using std::numeric_limits;
using std::ofstream;
using std::pow;
using std::ref;
using std::round;
using std::setprecision;
using std::setw;
using std::stod;
using std::streamsize;
using std::string;
using std::stringstream;
using std::swap;
using wmath::log2;
using wmath::mean_variance;
using wmath::popcount;


//template<typename EXP>
//const typename EXP::matrix_type inline matrix_sqrt(
//    const EXP& m
//    ){
//  return cholesky_decomposition(m).get_l;
//}

template<typename EXP>
const typename EXP::matrix_type inline matrix_sqrt(
    const matrix_op<op_make_symmetric<EXP>>& m
    ){
  eigenvalue_decomposition<matrix_op<op_make_symmetric<EXP>>> e(m);
  return e.get_pseudo_v()
    *diagm(dlib::sqrt(e.get_real_eigenvalues()))
    *inv(e.get_pseudo_v());
}

const matrix<double,3,1> inline get_orthogonal_vector(
    const matrix<double,3,1>& v){
  const auto nv = normalize(v);
  auto v0 = tmp(zeros_matrix(v));
  v0(0)=1;
  auto v1 = tmp(zeros_matrix(v));
  v0(1)=1;
  const double d0 = dot(v0,nv);
  const double d1 = dot(v0,nv);
  if (abs(d1)<abs(d0)) swap(v0,v1);
  return tmp(normalize(v0-dot(v0,nv)*nv));
}

double const inline area(
    const matrix<double,3,1>& v0,
    const matrix<double,3,1>& v1,
    const matrix<double,3,1>& v2){
  matrix<double,3,1> d0 = v1-v0;
  matrix<double,3,1> d1 = v2-v0;
  matrix<double,3,1> cross{ // cross product
      d0(1)*d1(2)-d0(2)*d1(1),
      d0(2)*d1(0)-d0(0)*d1(2),
      d0(0)*d1(1)-d0(1)*d1(0)};
  return 0.5*length(cross);
}

double const inline area(
    const matrix<double,3,1>& v0,
    const matrix<double,3,1>& v1,
    const matrix<double,3,1>& v2,
    const double& l // radius
    ){
  return area(v0,v1,v2); // TODO: fix spherical excess...
  const double a = acos(dot(v0,v1)*l*l)/l;
  const double b = acos(dot(v1,v2)*l*l)/l;
  const double c = acos(dot(v2,v0)*l*l)/l;
  const double s = (a+b+c)/2;
  const double e = 4*atan(sqrt(
        tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2)));
  //cerr << "spherical excess = " << e << endl;
  if (e<pi/100||isnan(e)) return area(v0,v1,v2);
  return e/(l*l);
}

/* Dk covariance contribution due to divergence
 * This models the incoming wave vector direction distribution.
 * It leads to a broadening of the observed peaks similar but discernable
 * from the reciprocal peak width.
 */
const matrix<double,3,3> S_divergence(
    const matrix<double,3,1>& win, // normed ingoing wave vector
    const double& div){
  return div*div*(identity_matrix<double>(3)-win*trans(win));
}

/* Dk covariance contribution due to dispersion aka bandwidth
 * given as the variance of the frequency i.e. σ(1/λ)
 */
const matrix<double,3,3> S_bandwidth(
    const matrix<double,3,1>& v,
    const double& bnd){
  return bnd*bnd*v*trans(v);
}

/* Dk covariance contribution due to detector response ( σ=1pix )
 * This models the "bleeding out" of pixel values into neighbouring pixels
 * and uncertainty in the location of the pixel.    
 * At the same time this enforces a minimum peak width of 1 pixel.
 * This enables to sample the predicted intensity or partiality on a
 * 1 pixel coarsity while having a maximum average relative sampling error
 * squared of less than 4e-9 on average
 */
const matrix<double,3,3> S_detector(
    const matrix<double,3,1>& k0, // corners of the
    const matrix<double,3,1>& k1, // pixel on the 
    const matrix<double,3,1>& k2, // detector in
    const matrix<double,3,1>& k3  // arbitrary order
    ){
  const matrix<double,3,1> nk0 = normalize(k0);
  const matrix<double,3,1> nk1 = normalize(k1);
  const matrix<double,3,1> nk2 = normalize(k2);
  const matrix<double,3,1> nk3 = normalize(k3);
  const matrix<double,3,1> c   = normalize(k0+k1+k2+k3);
  return (nk0-c)*trans(nk0-c)
        +(nk1-c)*trans(nk1-c)
        +(nk2-c)*trans(nk2-c)
        +(nk3-c)*trans(nk3-c);
}

/* The same as before but for a triangular pixel jaja haha why do
 * we need this? If we are doing oversampling of course.
 */
const matrix<double,3,3> S_detector(
    const matrix<double,3,1>& k0, // all the three
    const matrix<double,3,1>& k1, // vertices of the
    const matrix<double,3,1>& k2  // triangular shape
    ){
  const matrix<double,3,1> nk0 = normalize(k0);
  const matrix<double,3,1> nk1 = normalize(k1);
  const matrix<double,3,1> nk2 = normalize(k2);
  const matrix<double,3,1> c   = normalize(k0+k1+k2);
  return (nk0-c)*trans(nk0-c)
        +(nk1-c)*trans(nk1-c)
        +(nk2-c)*trans(nk2-c);
}

/* reciprocal peak shape contribution due to inherent width
 * this makes the observed peaks broader, similar but discernable from the
 * influence of beam divergence.
 * This approximates the reciprocal peak shape by a spherical gaussian -
 * which is probably a poor approximation... please make up your own sum of
 * elliptical shapes if this contribution is significant. 
 */
const matrix<double,3,3> P_peakwidth(
    const double& rpw
    ){
  const double rpw2 = rpw*rpw;
  return matrix<double,3,3>{rpw2,0,0,0,rpw2,0,0,0,rpw2};
}

/* reciprocal peak shape contribution due to mosaicity
 * This leads to a radial smearing of the reciprocal peaks orthogonal to
 * the hkl vector and to an radial smearing of the observed peaks.
 * On the detector this leads to a radial smearing of the peaks.
 */
matrix<double,3,3> P_mosaicity(
    const matrix<double,3,1> x,
    const double& mosaicity
    ){
  const double m2 = mosaicity*mosaicity;
  const double nx2 = length_squared(x);
  return m2*(nx2*identity_matrix<double>(3)-x*trans(x));
}

/* strain parametrizes the variance of the distribution of different strain
 * in the unit cells of the crystal. This leads to radially elongation of the
 * reciprocal peaks and consequently to an elongation of the observed,
 * virtually indistinguishable from the elongation due to wavelength bandwith.
 * The two parameters are degenerate for all practical purposes.
 */
matrix<double,3,3> P_strain(
    const matrix<double,3,1> x,
    const double& strain
    ){
  return (strain*x)*trans(strain*x);
}

double const inline norm_predict( // multiplicative normalization constant
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  return 1/sqrt(det(2*pi*(S)));
}

double const inline log_predict_exp( // exponential component of predict
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  return -0.5*trans(Dk-R*hkl)*inv(S)*(Dk-R*hkl);
}

double const inline predict(
    const matrix<double,3,1>& Dk, // incoming vector
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const double& i,                          // intensity
    const double& b,                          // B-Factor
    const matrix<double,3,3>& S   // compounded covariance matrix
    ); // Forward declaration


double const inline predict(
    const matrix<double,3,1>& Dk, // incoming vector
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const double& i,                          // intensity
    const double& b,                          // B-Factor
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  if (-0.5*b*length(Dk)+log_predict_exp(Dk,hkl,R,S)<600) return 0;
  return i*exp(-0.5*b*length(Dk)+log_predict_exp(Dk,hkl,R,S))
    *norm_predict(S);
}

/*
template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,3>& U,  // unit cell matrix
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  const auto   hkl = round(U*Dk);
  const int32_t h = int(hkl(0));
  const int32_t k = int(hkl(1));
  const int32_t l = int(hkl(2));
  const double   i = intensity.count(h,k,l)?intensity.at(h,k,l):0.0;
  const auto     R = inv(U);
  return predict(Dk,hkl,R,i,b,S);
}
*/

/*
template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& out,// outgoing vector
    const matrix<double,3,3>& U,  // unit cell matrix
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos                         // mosaicity
    ){
  const matrix<double,3,1> win  = normalize(in);
  const matrix<double,3,1> wout = normalize(out);
  const matrix<double,3,1> Dw   = win-wout;
  const matrix<double,3,1> Dk   = Dw*frq;
  const matrix<double,3,1> dhkl = U*Dk;
  const matrix<double,3,1> hkl  = round(dhkl);
  const matrix<double,3,3> R    = inv(U);
  const size_t r = reduce_encode(int(hkl(0)),int(hkl(1)),int(hkl(2)),8);
  const double i = intensity.count(r)?intensity.at(r):0.0;
  return predict(
      Dk,
      hkl,
      R,
      i,
      b,
      get_S_dsp(Dw,dsp)+get_S_div(win,div)+get_S_mos(mos));
}
*/

template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,3>&  U, // unit cell matrix
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos                         // mosaicity
    ){
  const matrix<double,3,1> c = (v0+v1+v2); //centroid
  return area(normalize(v0),normalize(v1),normalize(v2),1)
    *predict(in,c,U,intensity,b,frq,dsp,div,mos);
}

double const inline norm_predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,3>& S,  // wave vector covariance
    const matrix<double,3,1>& x,  // reciprocal peak vector = U*hkl
    const matrix<double,3,3>& P   // reciprocal peak shape
    ){
  return sqrt(det(2*pi*(S+P)));
}

double const inline nlog_partiality(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,3>& S,  // wave vector covariance
    const matrix<double,3,1>& x,  // reciprocal peak vector = U*hkl
    const matrix<double,3,3>& P   // reciprocal peak shape
    ){
  return 0.5*trans(Dk-x)*inv(S+P)*(Dk-x);
}

double const inline predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,3>& S,  // wave vector covariance
    const matrix<double,3,1>& x,  // reciprocal peak vector =U¯¹*hkl
    const matrix<double,3,3>& P   // reciprocal peak shape
    ){
  return exp(-nlog_partiality(Dk,S,x,P))/norm_predict(Dk,S,x,P);
}

/*
double const inline predict(
    const matrix<double,3,1>& win,// normalized incoming wave vector
    const matrix<double,3,1>& wot,// normalized outgoing wave vector
    const matrix<double,3,3>& U,  // unit cell matrix
    const matrix<double,3,3>& P,  // reciprocal peak shape
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  matrix<double,3,1> Dk = frq*(wot-win);
  return exp(-nlog_partiality(Dk,S,x,P))/norm_predict(Dk,S,x,P);
}
*/

/*
double const inline predict(
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,1>&  v0,// vertex 0 angular extent
    const matrix<double,3,1>&  v1,// vertex 1 angular extent
    const matrix<double,3,1>&  v2,// vertex 2 angular extent
    const matrix<double,3,1>&   x,
    const matrix<double,3,3>&   P // reciprocal peak shape
    ){
  const matrix<double,3,1> n0 = normalize(v0);
  const matrix<double,3,1> n1 = normalize(v1);
  const matrix<double,3,1> n2 = normalize(v2);
  return area(n0,n1,n2)*
    predict(normalize(in),normalize(n0+n1+n2),x,P);
}
*/

/*
double const inline predict(
    const matrix<double,3,1>&  Dk,// impulse transfer
    const matrix<double,3,3>&   S,// Dk distribution
    const matrix<double,3,1>&   x,// reciprocal peak vector = U¯¹hkl
    const matrix<double,3,3>&   P // reciprocal peak shape
    ){
  return exp(-nlog_partiality(Dk,S,x,P))/norm_predict(S,P); 
}
*/

/*
double const inline predict(
    const matrix<double,3,1>& win,// normed incoming wave vector
    const matrix<double,3,1>& wot,// normed outoing wave vector
    const double& frq,                        // 1/wavelength = frequency
    const matrix<double,3,3>&   S,// Dk distribution
    const matrix<double,3,1>&   x,// reciprocal peak vector = U¯¹hkl
    const matrix<double,3,3>&   P,// reciprocal peak shape
    const double& b                           // B-Factor
    ){
  const matrix<double,3,1> Dk  = frq*(wot-win);
  return predict(Dk,S,x,P,b);
}
*/

/*
double const inline predict(
    const matrix<double,3,1>& win,// normed incoming wave vector
    const matrix<double,3,1>&  n0,// normed vertex 0 angular extent
    const matrix<double,3,1>&  n1,// normed vertex 1 angular extent
    const matrix<double,3,1>&  n2,// normed vertex 2 angular extent
    const matrix<double,3,3>&   U,// unit cell matrix
    const matrix<double,3,3>& hkl,// Miller index
    const matrix<double,3,3>&   P,// reciprocal peak shape
    const matrix<double,3,3>&   S,// Dk distribution
    const double& b,                          // B-Factor
    const double& frq
    ){
  return area(n0,n1,n2)
        *predict(win,normalize(n0+n1+n2),frq,S,x,P);
}
*/
/*
double const inline predict(
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,1>&  v0,// vertex 0 angular extent
    const matrix<double,3,1>&  v1,// vertex 1 angular extent
    const matrix<double,3,1>&  v2,// vertex 2 angular extent
    const matrix<double,3,1>&  v3,// vertex 3 angular extent
    const matrix<double,3,3>&   U,// unit cell matrix
    const matrix<double,3,3>& hkl,// Miller index
    const matrix<double,3,3>&  P0,// reciprocal peak shape
    const double& b,              // B-Factor
    const double& frq,            // 1/wavelength = frequency
    const double& dsp,            // dispersion aka bandwidth
    const double& div,            // divergence
    const double& mos,            // mosaicity
    const double& str             // strain distribution on crystal
    ){
  const matrix<double,3,1> x  = U*hkl;
  const matrix<double,3,1> n0 = normalize(v0);
  const matrix<double,3,1> n1 = normalize(v1);
  const matrix<double,3,1> n2 = normalize(v2);
  const matrix<double,3,1> n3 = normalize(v3);
  const matrix<double,3,1> P  = P0+P_mosaicity(x,mos)+P_strain(x,str);
  return 0.25*(predict(in,n0,n1,n2,x,P,frq,dsp,div)
              +predict(in,n0,n1,n3,x,P,frq,dsp,div)
              +predict(in,n0,n2,n3,x,P,frq,dsp,div)
              +predict(in,n1,n2,n3,x,P,frq,dsp,div));
}
*/

/*
// solve  min( || B x - t || ) for x ∈ ℤ^n
const void inline closest_lattice_vector(
    const EXP1 B,
    const EXP2 x,
    const EXP3 t
    ){ // TODO
  return;
}

// this reduces to the closest lattice vector problem, на здоровье
const inline void closest_index(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,1>& v3, // vertex 3 angular extent
    const matrix<double,3,3>&  U, // unit cell matrix
    const matrix<double,3,3>&  R, // reciprocal unit cell
    const matrix<double,3,3>&  P, // inherent reciprocal peak shape
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    int32_t& h, int32_t& k, int32_t& l        // miller index to be set
    ){
  // TODO
  return;
}
*/

/*
const inline void closest_index(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,1>& v3, // vertex 3 angular extent
    const matrix<double,3,3>&  U, // unit cell matrix
    const matrix<double,3,3>&  R, // reciprocal unit cell
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str,                        // strain distribution on crystal
    int32_t& h, int32_t& k, int32_t& l        // miller index to be set
    ){
  const matrix<double,3,3> P  = P_peakwidth(rpw);
  closest_index(in,v0,v1,v2,v3,U,R,P,frq,dsp,div,mos,str);
}
*/

// helper function, nothing to see here
int digit_to_balanced_ternary(size_t i){
  if (i%3==0) return -1;
  if (i%3==1) return  0;
  if (i%3==2) return  1;
  return 0;
}

// iterate through h k l iteratively expanding the "radius"
inline void next_hkl_expansion(
    const int32_t& h0,const int32_t& k0,const int32_t& l0,
    int32_t& h,int32_t& k,int32_t& l,
    size_t& i){
  if (i==0){
    ++i;
    return;
  }
  const size_t r0 = i%(3*3*3)/(3*3);
  const size_t r1 = (i%(3*3*3)-r0)/3;
  const size_t r2 = (i%(3*3*3)-r0-r1);
  h = h0+(i/(3*3*3)+1)*digit_to_balanced_ternary(r0);
  k = k0+(i/(3*3*3)+1)*digit_to_balanced_ternary(r1);
  k = l0+(i/(3*3*3)+1)*digit_to_balanced_ternary(r2);
  ++i;
  if (i%(3*3*3)==0) ++i;
}

/* distance between wave vector difference and reciprocal peak vector given
 * wave vector covariance and reciprocal peak shape */
/*
double const inline nlog_partiality(
    const matrix<double,3,1>&  Dk,// impulse change
    const matrix<double,3,3>&   S,// wave vector covariance
    const matrix<double,3,1>&   x,// miller index  
    const matrix<double,3,3>&   P // reciprocal peak shape
    ){
  const matrix<double,3,1> d = Dk-x;
  return trans(d)*inv(P+S)*d;
}
*/

/* predict helper function with the total Dk distribution is computed, the
 * reciprocal peakshape has not jet been computed from mos and str */
/*
template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>&  Dk,// impulse transfer
    const matrix<double,3,3>&   S,// Dk distribution
    const matrix<double,3,1>&   U,// unit cell
    const matrix<double,3,1>&   R,// reciprocal unit cell
    const matrix<double,3,3>&  P0,// reciprocal peakshape 
    const refintmap& intensity,               // refererence intensities
    const double& b,                          // B-Factor
    const double& mos,
    const double& str,
    bool expand = false // default is correct if peaks have no overlap
    ){
  int32_t h0,k0,l0,h,k,l;
  closest_index(Dk,U,h0,k0,l0);
  size_t i=0;
  double p=0;
  while(true){
    next_hkl_expansion(h0,k0,l0,h,k,l,i++);
    const matrix<double,3,1> hkl{double(h),double(k),double(l)};
    const matrix<double,3,1> x = R*hkl;
    const matrix<double,3,3> P= P0+P_strain(x,str)+P_mosaicity(x,mos);
    const double d = nlog_partiality(Dk,S,x,P);
    if ((d<6)&&(i>0)) expand=true;
    p += intensity(h,k,l)*exp(-0.5*b*length_sqared(x)-d)/norm_predict(S,P);
    if ((i%(3*3*3)==26)&&(expand==false)) break;
    else expand=false;
  }
  return exp(-nlog_partiality(Dk,S,x,P))/norm_predict(S,P);
}
*/

/* secondary prediction function given reference, unit cell,
 * the inherent reciprocal peakshape and other reciprocal peakshape parameters
 * */
/*
template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>& win,// incoming vector
    const matrix<double,3,1>& w0, // vertex 0 angular extent
    const matrix<double,3,1>& w1, // vertex 1 angular extent
    const matrix<double,3,1>& w2, // vertex 2 angular extent
    const matrix<double,3,3>&  U, // unit cell matrix
    const matrix<double,3,3>&  R, // reciprocal unit cell
    const matrix<double,3,3>&  P, // reciprocal peak shape
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  const matrix<double,3,1> c  = normalize(w0+w1+w2);
  const matrix<double,3,1> Dw = frq*(c-win);
  const matrix<double,3,3> S  = S_bandwidth(Dw,dsp)
                               +S_divergence(win,div)
                               +S_detector(w0,w1,w2);
  return area(w0,w1,w2)*predict(frq*Dw,S,x,P,rpw,mos,str);
}
*/

/* main prediction function given reference, unit cell
 * and reciprocal peakshape parameters */
template<typename refintmap>
double const inline predict(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,1>& v3, // vertex 3 angular extent
    const matrix<double,3,3>&  U, // unit cell matrix
    const matrix<double,3,3>&  R, // reciprocal unit cell
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str,                        // strain distribution on crystal
    bool expand = false
    ){
  const matrix<double,3,1> win= normalize(in);
  const matrix<double,3,1> w0 = normalize(v0);
  const matrix<double,3,1> w1 = normalize(v1);
  const matrix<double,3,1> w2 = normalize(v2);
  const matrix<double,3,1> w3 = normalize(v3);
  const matrix<double,3,3> P  = P_peakwidth(rpw);
  return 0.25*(
      predict(win,w0,w1,w2,U,R,P,intensity,b,frq,dsp,div,mos,str,expand)
     +predict(win,w1,w2,w3,U,R,P,intensity,b,frq,dsp,div,mos,str,expand)
     +predict(win,w0,w2,w3,U,R,P,intensity,b,frq,dsp,div,mos,str,expand)
     +predict(win,w0,w1,w3,U,R,P,intensity,b,frq,dsp,div,mos,str,expand));
}

/* Partiality function given ingoing wave direction, pixel corners,
 * reciprocal peak shape, frq dsp div rpw mos str */
double const inline nlog_partiality(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,1>& v3, // vertex 3 angular extent
    const matrix<double,3,1>&  x, // reciprocal peak vector = U¯¹hkl
    const matrix<double,3,3>&  P, // reciprocal peak shape
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    const double& str,                        // strain distribution on crystal
    bool expand = false
    ){
  const matrix<double,3,1> win= normalize(in);
  const matrix<double,3,1> w0 = normalize(v0);
  const matrix<double,3,1> w1 = normalize(v1);
  const matrix<double,3,1> w2 = normalize(v2);
  const matrix<double,3,1> w3 = normalize(v3);
  const matrix<double,3,1> c  = normalize(v0+v1+v2+v3);
  const matrix<double,3,1> Dw = c-win;
  return nlog_partiality(frq*Dw,
      S_bandwidth(Dw,dsp)+S_divergence(win,div)+S_detector(w0,w1,w2,w3),
      x,P+P_mosaicity(x,str)+P_strain(x,str));
}

/* prediction function given miller index,
 * reciprocal peak shape, frq, dsp div rpw mos str */
double const inline predict(
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,3>&   R,// reciprocal unit cell
    const matrix<double,3,1>& hkl,// miller index
    const matrix<double,3,3>&   P,// reciprocal peak shape
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  const matrix<double,3,1> win = normalize(in);
  const matrix<double,3,1> kin = frq*win;
  const matrix<double,3,1>   x = R*hkl;
  matrix<double,3,1> Dk = frq*normalize(x);
  matrix<double,3,1> k  = Dk+kin;
  matrix<double,3,1> Dw;
  matrix<double,3,3> S;
  for (size_t i=0;i!=2;++i){
    Dw = normalize(Dk);
    S = S_divergence(win,div)
       +S_bandwidth(Dw,dsp)
       +P
       +P_mosaicity(x,mos)
       +P_strain(x,str);
    k = frq*normalize(
        inv(inv(S)-Dw*trans(Dw)*inv(S)+Dw*trans(Dw))
         *((inv(S)-Dw*trans(Dw)*inv(S))*(x+kin)+Dw*trans(Dw)*k));
    Dk = k-kin;
  }
  const double d = trans(Dk-R*hkl)*Dw*trans(Dw)*inv(S)*Dw*trans(Dw)*(Dk-R*hkl);
  if (0.5*d>=LOG_DBL_MAX) return 0;
  return exp(-0.5*d)/sqrt(2*pi*(trans(Dw)*S*Dw));
}

/* prediction function given miller index,
 * reciprocal peak shape, frq, dsp div rpw mos str */
double const inline predict(
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,3>&   R,// reciprocal unit cell
    const matrix<double,3,1>& hkl,// miller index
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // inherent reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  return predict(in,R,hkl,P_peakwidth(rpw),frq,dsp,div,mos,str);
}

/*Partiality helper functor to optimize for kout given
 * win, kout, R, hkl, P, frq, dsp, div, mos, str
 */
/*
struct nlogpartial_functor{
  const matrix<double,3,1>& win;
  const matrix<double,3,1>&   x;
  const matrix<double,3,3>&   P;// reciprocal peak shape
  const double& frq;
  const double& dsp;                        // dispersion
  const double& div;                        // divergence
  const double& mos;                        // mosaicity
  const double& str;                        // strain distribution on crystal
  const double operator()(const matrix<double,3,1>& _k) const {
    const double _frq = length(_k);
    const matrix<double,3,1> w = normalize(_k);
    const matrix<double,3,1> k = frq*w;
    //const matrix<double,3,1> k = frq*w;
    //const matrix<double,3,1> k = x;
    const matrix<double,3,1> Dw = w-win;
    const matrix<double,3,1> Dk = frq*Dw;
    const matrix<double,3,3> S = S_divergence(win,div)
                                //+S_divergence(w,1.0/16)
                                +S_bandwidth(frq*normalize(Dw),dsp)
                                +P
                                +P_mosaicity(x,mos)
                                +P_strain(x,str); 
    
    const double la = abs(0.5*trans(Dk-x)*inv(S)*(Dk-x))
                     +abs(0.5*(_frq-frq)*(_frq-frq)/(dsp*dsp*frq*frq));
    return la;
    //const double a = la<LOG_DBL_MAX?exp(-la):0;
    //return exp(-0.5*(_frq-frq)*(_frq-frq)/(dsp*dsp))*a;
  }
};
*/
struct nlogpartial_functor{
  const matrix<double,3,1>& win;
  const matrix<double,3,1>&   m;// R*hkl
  const matrix<double,3,3>&   P;// reciprocal peak shape
  const double& f;                          // frequency
  const double& b;                          // bandwidth
  const double& d;                          // divergence
  const double& mos;                        // mosaicity
  const double& str;                        // strain distribution on crystal
  const double operator()(const matrix<double,3,1>& x) const {
    const double length_x = length(x);
    const matrix<double,3,1> w = x/length_x;
    const matrix<double,3,1> Dw= w-win;
    const matrix<double,3,3> S = S_divergence(win,d)
                                +S_bandwidth(Dw,b)
                                +P
                                +P_mosaicity(m,mos)
                                +P_strain(m,str);
    // increase numerical stability of inversion by taking square root
    // const matrix<double,3,3> iS12 = inv(matrix_sqrt(make_symmetric(S)));
    const double b2 = b*b;
    const matrix<double,3,1> t0 = f*w-f*win-m;
    return 0.5*trans(t0)*inv(S)*t0
          +0.5*(length_x-f)*(length_x-f)/b2;
  }
  const matrix<double,3,1> derivative(
      const matrix<double,3,1>& x
      ) const {
    const double length_x = length(x);
    const matrix<double,3,1> w = x/length_x;
    const matrix<double,3,1> Dw= w-win;
    const matrix<double,3,3> S = S_divergence(win,d)
                                +S_bandwidth(Dw,b)
                                +P
                                +P_mosaicity(m,mos)
                                +P_strain(m,str);
    const matrix<double,3,3> iS = inv(S);
    const double b2 = b*b;
    const matrix<double,3,1> t0 = f*w-f*win-m;
    const double t1 = trans(t0)*inv(S)*Dw;
    const double t2 = trans(w)*iS*t0;
    return f/length_x*iS*t0 // main component
      +t1*b2/length_x*(t2*w-iS*t0)-f/length_x*t2*w // narly crossterms
      +(1-f/length_x)/b2*x; // frequency offset to force kout to have frq norm
  }
};

/* Partiality function given miller index,
 * reciprocal peak shape, frq, dsp div rpw mos str */
double const inline partial(
          matrix<double,3,1>&   k,
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,3>&   R,// reciprocal unit cell
    const matrix<double,3,1>& hkl,// miller index
    const matrix<double,3,3>&   P,// reciprocal peak shape
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // bandwidth
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  //cout << "partial " << trans(hkl);
  const matrix<double,3,1> win = normalize(in);
  const matrix<double,3,1>   x = R*hkl;
        matrix<double,3,1> Dk  = x;
  //cout << "intial Dk= " << trans(Dk);
  k = Dk+frq*win; // kout - kin = Dk
  //k = frq*normalize(k);
  //Dk = k-kin;
  //cout << "second Dk= " << trans(Dk);
  nlogpartial_functor target{win,x,P,frq,dsp,div,mos,str};
  //cout << "initial guess for k is:" << endl;
  //cout << trans(k);
  //cout << trans(target.derivative(k));
  //cout << target(k) << endl;
  //cout << "derivatives:" << endl;
  dlib::find_min(
      dlib::bfgs_search_strategy(),
      dlib::objective_delta_stop_strategy(1e-8,256),
      target,
      [&target](const matrix<double,3,1>& x){return target.derivative(x);},
      k,
      0);
  
  //cout << trans(k);
  //cout << trans(Dk);
  //cout << target(k) << endl;

  const matrix<double,3,1> w = normalize(k);
  //Dk = frq*(w-win);
  const double _frq = length(k);
  Dk = k-_frq*win;
  const matrix<double,3,3> S = S_divergence(win,div)
                              +S_bandwidth(frq*normalize(Dk),dsp)
                              +P
                              +P_mosaicity(x,mos)
                              +P_strain(x,str);
  //const double la = 0.5*length(
  //      inv(cholesky_decomposition<decltype(S)>(S).get_l())*w*trans(w)*(Dk-x));
  //const double la = abs(0.5*trans(w*trans(w)*(Dk-x))*inv(S)
  //                               *w*trans(w)*(Dk-x));
  k = frq*w;
  const double la = target(k);
                   //+(_frq-frq)*(_frq-frq)/(frq*frq*dsp*dsp);
  if (abs(la)>LOG_DBL_MAX) return 0;
  return exp(-la);
}

/* Partiality function given miller index,
 * reciprocal peak shape, frq, dsp div rpw mos str */
double const inline partial(
          matrix<double,3,1>&   k,
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,3>&   R,// reciprocal unit cell
    const matrix<double,3,1>& hkl,// miller index
    const double& frq,                        // 1/wavelength = frequency
    const double& bnd,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // inherent reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  return partial(k,in,R,hkl,P_peakwidth(rpw),frq,bnd,div,mos,str);
}

/*
double const inline partial(
    const matrix<double,3,1>&   k,
    const matrix<double,3,1>&  in,// incoming vector
    const matrix<double,3,3>&   R,// reciprocal unit cell
    const matrix<double,3,1>& hkl,// miller index
    const double& frq,                        // 1/wavelength = frequency
    const double& bnd,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // inherent reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  Dk = k-_frq*win;
  const matrix<double,3,3> S = S_divergence(win,div)
                              +S_bandwidth(frq*normalize(Dk),dsp)
                              +P
                              +P_mosaicity(x,mos)
                              +P_strain(x,str);
  const matrix<double,3,1> x = R*hkl;
  return trans(Dk-x)*inv(S)*(Dk-x)/det(2*pi*S);
}
*/

/* Main partiality function given ingoing wave direction, pixel corners,
 * frq dsp div rpw mos str */
double const inline nlog_partiality(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& v0, // vertex 0 angular extent
    const matrix<double,3,1>& v1, // vertex 1 angular extent
    const matrix<double,3,1>& v2, // vertex 2 angular extent
    const matrix<double,3,1>& v3, // vertex 3 angular extent
    const matrix<double,3,1>&  x, // reciprocal peak vector = U¯¹hkl
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& rpw,                        // reciprocal peak width
    const double& mos,                        // mosaicity
    const double& str                         // strain distribution on crystal
    ){
  return nlog_partiality(
      in,v0,v1,v2,v3,x,P_peakwidth(rpw),frq,dsp,div,mos,str);
}

/*namespace derivative{
double const inline diff_predict(
    const matrix<double,3,1>& in, // incoming vector
    const matrix<double,3,1>& out,// outgoing vector
    const matrix<double,3,3>& U,  // unit cell matrix
    const refintmap& intensity,               // reference intensities
    const double& b,                          // B-Factor
    const double& frq,                        // 1/wavelength = frequency
    const double& dsp,                        // dispersion
    const double& div,                        // divergence
    const double& mos,                        // mosaicity
    matrix<double,3,3>& dU,                   // derivative w.r.t. unit cell 
    double& db,                               // derivative w.r.t. B-Factor
    double& ddsp,                             // derivative w.r.t. dispersion
    double& ddiv,                             // derivative w.r.t. divergence
    double& dmos                              // derivative w.r.t. mosaicity
    ){
  const matrix<double,3,1> win  = normalize(in);  // ingoing wave vector
  const matrix<double,3,1> wout = normalize(out); // outgoing wave vector
  const matrix<double,3,1> Dw   = win-wout;       // wavevector difference
  const matrix<double,3,1> Dk   = Dw*frq;         // impulse change
  const matrix<double,3,1> hkl  = round(U*Dk);    // closest miller index
  const matrix<double,3,3> R    = inv(U);         // reciprocal unit cell
  const size_t r = reduce_encode(int(hkl(0)),int(hkl(1)),int(hkl(2)),8);
  const double i = intensity.count(r)?intensity.at(r):0.0;
  const matrix<double,3,3> S_dsp = get_S_dsp(Dw,dsp);
  const matrix<double,3,3> S_div = get_S_div(Dw,div);
  const matrix<double,3,3> S_mos = get_S_mos(Dw,mos);
  const matrix<double,3,3> S     = S_dsp+S_div+S_mos;
  db  += -0.5*length(Dk)*predict(Dk,hkl,R,i,b,S);
  ddsp+= 2*dsp*sum(pointwise_multiply(S_dsp,diffS_predict(Dk,hkl,R,i,b,S)));
  ddiv+= 2*div*sum(pointwise_multiply(S_div,diffS_predict(Dk,hkl,R,i,b,S)));
  dmos+= 2*mos*sum(pointwise_multiply(S_mos,diffS_predict(Dk,hkl,R,i,b,S)));
  return predict(Dk,hkl,R,i,b,S);
}

matrix<double,3,3> const inline diffR_predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  return predict(Dk,hkl,R,i,b,S)*diffR_log_predict(Dk,hkl,R,S);
}

double const inline diffb_predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const double& i,                          // intensity
    const double& b,                          // B-Factor
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  return -0.5*length(Dk)*predict(Dk,hkl,R,i,b,S);
}

matrix<double,3,3> const inline diffS_predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const double& i,            // intensity
    const double& b,            // B-Factor
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  const auto R = inv(U);
  return predict(Dk,hkl,R,i,b,S)*diffS_log_predict(Dk,hkl,R,S);
}

matrix<double,3> diff_norm_predict( // multiplicative normalization constant
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  // d(det(M)) = det(m)*Tr(m¯¹dm)
  // d(1/sqrt(f(x))) = -(d(f(x))/dx)/(2*sqrt(f(x)³))
  // d(1/sqrt(det(2*pi*S))) = 
  // det(m)*Tr(m¯¹dm)/(2*sqrt(det(2*pi*S)*det(2*pi*S)*det(2*pi*S)))
  return det(S)*trace(inv(S)*)/(2*sqrt(det(2*pi*S)*det(2*pi*S)*det(2*pi*S)));
}

matrix<double,3,3> const inline diffR_log_predict_exp(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  // diff(-0.5*trans(x-μ)*inv(S)*(x-μ),μ) =
  // inv(S) * (x-μ)
  // diff(-0.5*trans(Dk-R*hkl)*inv(S)*(Dk-R*hkl),R) = 
  // diff(R*hkl,R) * inv(S) * (Dk-R*hkl) =
  // trans(hkl) * inv(S) * (Dk-R*hkl)
  return trans(hkl)*inv(S)*(Dk-R*hkl);
}

matrix<double,3,3> const inline diffS_log_predict(
    const matrix<double,3,1>& Dk, // impulse change
    const matrix<double,3,1>& hkl,// Miller Index
    const matrix<double,3,3>& R,  // reciprocal unit cell matrix
    const matrix<double,3,3>& S   // compounded covariance matrix
    ){
  return -0.5*(inv(S)-inv(S)*(Dk-R*hkl)*trans(Dk-R*hkl)*inv(S));
}
}*/
}
