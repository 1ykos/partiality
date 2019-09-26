#ifndef PARTIALITY_H
#define PARTIALITY_H
#include "wmath.hpp"
#include "encode.hpp"
#include "geometry.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

namespace partiality{
  constexpr double pi          = 3.14159265358979323846;
  constexpr double LOG_DBL_MAX = 709.78271289338402993962517939506;
  using SYMMETRY::decode;
  using SYMMETRY::reduce;
  using SYMMETRY::reduce_encode;
  using dlib::abs;
  using dlib::cholesky_decomposition;
  using dlib::chol;
  using dlib::diag;
  using dlib::diagm;
  using dlib::dot;
  using dlib::eigenvalue_decomposition;
  using dlib::identity_matrix;
  using dlib::inv;
  using dlib::inv_lower_triangular;
  using dlib::pinv;
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
  using dlib::tmp;
  using dlib::trace;
  using dlib::zeros_matrix;
  using geometry::panel;
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
  using std::function;
  using std::ifstream;
  using std::isnan;
  using std::istream;
  using std::map;
  using std::max_element;
  using std::minmax;
  using std::mt19937_64;
  using std::normal_distribution;
  using std::numeric_limits;
  using std::ofstream;
  using std::random_device;
  using std::ref;
  using std::round;
  using std::setprecision;
  using std::setw;
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
  using wmath::bswap;
  using wmath::inverf;
  using wmath::log2;
  using wmath::mean_variance;
  using wmath::popcount;
  using wmath::pow;
  using wmath::rol;
  using wmath::circadd;
  using wmath::digits;
  using wmath::hash_functor;

  struct prediction_ellipse{
    const matrix<double,2,1> m;   // center of ellipse
    const matrix<double,2,2> S;   // covariance
    const matrix<double,2,2> iS;  // inverse covariance
    const double a2;              // partiality = exp(-0.5*a2);
    const double lorentz;         // lorentz factor
    double inline operator()(const double& fs,const double& ss) const {
      const matrix<double,2,1> x{fs,ss};
      return trans(x-m)*iS*(x-m);    // this is for shape
    }
    double inline pred(const double fs,const double& ss) const {
      return lorentz*exp(-0.5*(operator()(fs,ss)+a2))/sqrt(4*pi*pi*det(S));
    }
    double inline part(const double& fs,const double& ss) const {
      return exp(-0.5*(operator()(fs,ss)+a2))*sqrt(det(iS))/(2*pi);
    }
    double inline proj(const double fs,const double& ss) const {
      return exp(-0.5*(operator()(fs,ss)));
    }
    double inline extent_fs() const {
      return sqrt(S(0,0));
      //return sqrt(S(0,0)-a2/2>0?S(0,0)-a2/2:0);
    }
    double inline extent_ss() const {
      return sqrt(S(1,1));
      //return sqrt(S(1,1)-a2/2>0?S(1,1)-a2/2:0);
    }
    double inline part() const {
      return 2*exp(-0.5*a2)/pi;
    }
    double inline frac() const {
      return lorentz*exp(-0.5*a2);
    }
  };

  tuple<matrix<double,2,1>,matrix<double,2,2>>
  const inline approximate(const vector<prediction_ellipse>& ellipses){
    double sumw = 0;
    matrix<double,2,1> mean{0,0,0};
    matrix<double,2,2> M2{0,0,0,0};
    matrix<double,2,2> M3{0,0,0,0};
    for (auto it=ellipses.begin();it!=ellipses.end();++it){
      const double w    = it->frac();
      const matrix<double,2,1> m = it->m;
      const matrix<double,2,2> S = it->S;
      const double temp = w+sumw;
      const matrix<double,2,1> delta = m-mean;
      const matrix<double,2,1> R     = delta*w/temp;
      mean   += R;
      M2     += delta*trans(delta)*sumw*w/temp;
      sumw   = temp;
      M3     += w*S;
    }
    return {mean,(M3+M2)/sumw};
  }
  
  /* Dk covariance contribution due to divergence
   * This models the incoming wave vector direction distribution.
   * It leads to a broadening of the observed peaks similar but discernable
   * from the reciprocal peak width.
   */
  matrix<double,3,3> inline S_divergence(
      const matrix<double,3,1>& win, // normed ingoing wave vector
      const double& div){
    return pow(div,2u)*(identity_matrix<double>(3)-win*trans(win));
  }

  /* Dk covariance contribution due to dispersion aka bandwidth
   * given as the variance of the wavenumber i.e. σ(1/λ)
   */
  matrix<double,3,3> inline S_bandwidth(
      const matrix<double,3,1>& v,
      const double& bnd){
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
      ){
    const double rpw2 = pow(rpw,2u);
    return matrix<double,3,3>{rpw2,0,0,0,rpw2,0,0,0,rpw2};
  }
  
  /* reciprocal peak shape contribution due to mosaicity
   * This leads to a radial smearing of the reciprocal peaks orthogonal to
   * the hkl vector and to an radial smearing of the observed peaks.
   * On the detector this leads to a radial smearing of the peaks.
   */
  matrix<double,3,3> inline P_mosaicity(
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
  matrix<double,3,3> inline P_strain(
      const matrix<double,3,1> x,
      const double& strain
      ){
    return (strain*x)*trans(strain*x);
  }
  
  struct source_summand{
    matrix<double,3,1> win;
    double wvn; // wavenumber, reiprocal wavelength
    double bnd; // bandwidth, in wavenumbers
    matrix<double,3,3> div;
  };

  struct crystl_summand{
    matrix<double,3,3> U;    // unit cell
    matrix<double,3,3> R;    // reciprocal unit cell { inv(U) }
    double mosaicity;
    matrix<double,3,3> peak;
    double strain;
    // 3x3x3x3 * 3x1 -> 3x3x3
  };
 
  [[deprecated]]
  matrix<double,3,3> const inline constant_covariance_matrix(
      const matrix<double,3,1>& win,
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const double frq = 1.0,
      const double bnd = 0.0,
      const double div = 0.0,
      const double rpw = 0.0,
      const double mos = 0.0,
      const double str = 0.0
      ){
    return frq*frq*S_divergence(win,div)
         + P_peakwidth(bnd)
         + P_peakwidth(rpw)
         + P_mosaicity(m,mos)
         + P_strain(m,str);
  }

  [[deprecated]]
  matrix<double,3,3> const inline covariance_matrix(
      const matrix<double,3,1>& win,
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const double frq = 1.0,
      const double bnd = 0.0,
      const double div = 0.0,
      const double rpw = 0.0,
      const double mos = 0.0,
      const double str = 0.0
      ){
      return frq*frq*S_divergence(win,div)
           + S_bandwidth(Dw,bnd)
           + P_peakwidth(rpw)
           + P_mosaicity(m,mos)
           + P_strain(m,str);
  }
  
  matrix<double,3,3> const inline constant_covariance_matrix(
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const source_summand& source,
      const crystl_summand& crystl
      ){
    return pow(source.wvn,2u)*source.div
          +pow(source.bnd,2u)*identity_matrix<double>(3) // isotropic
          +P_mosaicity(m,crystl.mosaicity)
          +P_strain(m,crystl.strain)
          +crystl.peak;
  }
  
  matrix<double,3,3> const inline matrix_S(
      const matrix<double,3,1>& Dw,
      const source_summand& source
      ){
    return   pow(source.wvn,2u)*source.div
            +S_bandwidth(normalize(Dw),source.bnd);
  }
  
  matrix<double,3,3> const inline matrix_P(
      const matrix<double,3,1>& m,
      const crystl_summand& crystl
      ){
    return   P_mosaicity(m,crystl.mosaicity)
            +P_strain(m,crystl.strain)
            +crystl.peak;
  }
  
  matrix<double,3,3> const inline covariance_matrix(
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const source_summand& source,
      const crystl_summand& crystl
      ){
    return   pow(source.wvn,2u)*source.div
            +S_bandwidth(normalize(Dw),source.bnd)  // anisotropic
            +P_mosaicity(m,crystl.mosaicity)
            +P_strain(m,crystl.strain)
            +crystl.peak;
    return  matrix_S(Dw,source)+matrix_P(m,crystl);
  }

  struct parameters{
    vector<tuple<double,source_summand>> source;
    vector<tuple<double,crystl_summand>> crystl;
    source_summand const inline average_source() const {
      double sumw = 0;
      matrix<double,3,1> mean_win{0,0,0};
      double mean_wvn = 0;
      matrix<double,3,3> M2_div{0,0,0,0,0,0,0,0,0};
      matrix<double,3,3> div{0,0,0,0,0,0,0,0,0};
      double M2_bnd = 0;
      double bnd = 0;
      for (auto it=source.begin();it!=source.end();++it){
        const double temp = get<0>(*it)+sumw;
        const matrix<double,3,1> delta_div = get<1>(*it).win-mean_win;
        const matrix<double,3,1> R_div     = delta_div*get<0>(*it)/temp;
        mean_win += R_div;
        M2_div   += delta_div*trans(delta_div)*sumw*get<0>(*it)/temp;
        div      += get<0>(*it)*get<1>(*it).div;
        const double delta_wvn = get<1>(*it).wvn-mean_wvn;
        const double R_wvn     = delta_wvn*get<0>(*it)/temp;
        mean_wvn += R_wvn;
        M2_bnd   += pow(delta_wvn,2u)
                    *sumw*get<0>(*it)/temp;
        bnd      += pow(get<1>(*it).bnd,2u);
        sumw      = temp;
      }
      mean_win = normalize(mean_win);
      bnd+=M2_bnd;
      bnd/=sumw;
      bnd=sqrt(bnd);
      div+=M2_div;
      div*=(identity_matrix<double>(3)-mean_win*trans(mean_win))/sumw;
      return source_summand{mean_win,mean_wvn,bnd,div};
    }
  };

  tuple<double,double> const inline minmaxq(
      const matrix<double,3,1> win,
      const geometry::geometry& geom
    ){
    double maxq=numeric_limits<double>::lowest();
    double minq=numeric_limits<double>::max();
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
      matrix<double,3,1> v0 = -(*it)(matrix<double,2,1>{0,0});
      matrix<double,3,1> v1 = -(*it)(matrix<double,2,1>{double(it->nfs),0});
      matrix<double,3,1> v2 = -(*it)(matrix<double,2,1>{0,double(it->nss)});
      matrix<double,3,1> v3 = -(*it)(matrix<double,2,1>{double(it->nfs),double(it->nss)});
      matrix<double,3,1> v01= normalize(v0-v1);
      matrix<double,3,1> v12= normalize(v1-v2);
      matrix<double,3,1> v23= normalize(v2-v3);
      matrix<double,3,1> v30= normalize(v3-v0);
      double q;
      // edges:
      q = length(win+normalize(v0)); minq=q<minq?q:minq; maxq=q>maxq?q:maxq;
      q = length(win+normalize(v1)); minq=q<minq?q:minq; maxq=q>maxq?q:maxq;
      q = length(win+normalize(v2)); minq=q<minq?q:minq; maxq=q>maxq?q:maxq;
      q = length(win+normalize(v3)); minq=q<minq?q:minq; maxq=q>maxq?q:maxq;
      // sides:
      matrix<double,2,1> m;
      matrix<double,3,1> p01 = double(trans(v0)*v01)*v01; m=(*it)(p01);
      if (m(0)>-1&&m(0)<it->nfs+1&&m(1)>-1&&m(1)<it->nss+1){
        q = length(normalize(p01-win)); minq=q<minq?q:minq;
      }
      matrix<double,3,1> p12 = double(trans(v1)*v12)*v12; m=(*it)(p01);
      if (m(0)>-1&&m(0)<it->nfs+1&&m(1)>-1&&m(1)<it->nss+1){
        q = length(normalize(p12-win)); minq=q<minq?q:minq;
      }
      matrix<double,3,1> p23 = double(trans(v2)*v23)*v23; m=(*it)(p01);
      if (m(0)>-1&&m(0)<it->nfs+1&&m(1)>-1&&m(1)<it->nss+1){
        q = length(normalize(p23-win)); minq=q<minq?q:minq;
      }
      matrix<double,3,1> p30 = double(trans(v3)*v30)*v30; m=(*it)(p01);
      if (m(0)>-1&&m(0)<it->nfs+1&&m(1)>-1&&m(1)<it->nss+1){
        q = length(normalize(p30-win)); minq=q<minq?q:minq;
      }
      // TODO: surfaces for finding minimum
    }
    return tuple<double,double>{minq,maxq};
  }

/*  tuple<double,double> const inline merge_gaussian_kernels(
    vector<gaussian_kernel> data){
    double sumw=0,mean=0,M2=0;
    for (auto it=data.begin();it!=data.end();++it){
      const double temp = it->w+sumw;
      const double delta = it->m-mean;
      const double R = delta*w/temp;
      mean += R;
      M2   += (delta*delta+it->s)*sumw*w/temp;
      sumw  = temp;
    }
    return tuple<double,double>{mean,sqrt(M2/sumw)};
  }
*/
  
  double inline estimate_max_f(
      const source_summand& parm,
      const double& min_partiality){
    return parm.wvn+sqrt(-log(min_partiality)*2*parm.bnd*parm.bnd);
  }

  matrix<int32_t,3,1> inline extent_hkl(
      const tuple<double,double>& qrange,
      const crystl_summand& parm,
      const double& maxf){
    const double maxq = get<1>(qrange);
    const int maxh =
      floor(
        maxq*maxf/sqrt(pow(parm.R(0,0),2u)
                      +pow(parm.R(1,0),2u)
                      +pow(parm.R(2,0),2u))
      );
    const int maxk =
      floor(
        maxq*maxf/sqrt(pow(parm.R(0,1),2u)
                      +pow(parm.R(1,1),2u)
                      +pow(parm.R(2,1),2u))
      );
    const int maxl =
      floor(
        maxq*maxf/sqrt(pow(parm.R(0,2),2u)
                      +pow(parm.R(1,2),2u)
                      +pow(parm.R(2,2),2u))
      );
    return matrix<int32_t,3,1>{maxh,maxk,maxl};
  }

  matrix<double,3,1> const inline optimize_wout(
      const matrix<double,3,1>& w,
      const matrix<double,3,1>& m,
      const source_summand& source,
      const crystl_summand& crystl,
      bool& success
      ){
    constexpr bool o = false;
    if (o) cout << "optimize_wout" << endl;
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
    if (o) cout << v << endl;
    for (double c=1.0;c>1e-4;c*=exp(-1)){
      if (o) cout << "search loop" << endl;
      const matrix<double,3,1> w1=normalize(w-c*Dx);
      const matrix<double,3,3> iS=inv(covariance_matrix(w1-win,m,source,crystl));
      const double _v = pow(wvn,2u)*trans(w1-m1)*iS*(w1-m1);
      if (o) cout << _v << " " << v << endl;
      if (o) cout << trans(w1);
      if (pow(wvn,2u)*trans(w1-m1)*iS*(w1-m1)<v) return w1;
    }
    if (o) cout << "not a minimizing direction :(" << endl;
    success = false;
    return w;
  }
  
  vector<
    tuple<
      uint64_t,
      vector<
        tuple<
          const geometry::panel*,
          vector<prediction_ellipse>
        >
      >
    >
  >
  const inline get_prediction_ellipses(
    const parameters& parm,
    const geometry::geometry& geom,
    const double fraction = 1.0/64,
    const size_t oversampling = 1
    ){
    vector<tuple<uint64_t,vector<tuple<
      const geometry::panel*,
      vector<prediction_ellipse>
    >>>> ellipses;
    source_summand average_source = parm.average_source();
    const matrix<double,3,1> kin = average_source.wvn*average_source.win;
    typedef tuple<int32_t,int32_t,int32_t> M;
    unordered_set<M,hash_functor<M>> hkls;
    const double max_e = -sqrt(2)*inverf(fraction);
    cout << max_e << endl;
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
      for (size_t ss=0;ss<=it->nss;++ss) for (size_t fs=0;fs<=it->nfs;++fs){
        for (auto crystl=parm.crystl.begin();crystl!=parm.crystl.end();++crystl){
          const matrix<double,2,1> x{double(fs+0.5),double(ss+0.5)};
          const matrix<double,3,1> y = (*it)(x);
          const matrix<double,3,1> Dw= normalize(y)-average_source.win;
          const matrix<double,3,1> m =
            get<1>(*crystl).U*Dw*average_source.wvn;
          //cout << trans(m);
          const matrix<double,3,3> S =
            covariance_matrix(
                Dw,
                m,
                average_source,
                get<1>(parm.crystl[0]));
          const double eh = max_e*sqrt(S(0,0));
          const double ek = max_e*sqrt(S(1,1));
          const double el = max_e*sqrt(S(2,2));
          const int32_t lo_h = round(m(0)-eh);//floor(m(0)-eh);
          const int32_t hi_h = round(m(0)+eh);// ceil(m(0)+eh);
          const int32_t lo_k = round(m(1)-ek);//floor(m(1)-ek);
          const int32_t hi_k = round(m(1)+ek);// ceil(m(1)+ek);
          const int32_t lo_l = round(m(2)-el);//floor(m(2)-el);
          const int32_t hi_l = round(m(2)+el);// ceil(m(2)+el);
          for(int32_t l=lo_l;l<=hi_l;++l)
          for(int32_t k=lo_k;k<=hi_k;++k)
          for(int32_t h=lo_h;h<=hi_h;++h) hkls.insert({h,k,l});
        }
      }
    }
    cout << "considering " << hkls.size() << " hkl indices" << endl;
    matrix<double,3,1>* ms = new matrix<double,3,1>[parm.crystl.size()]();
    matrix<double,3,1>* wouts =
      new matrix<double,3,1>[parm.source.size()*parm.crystl.size()]();
    double max_intensity     = numeric_limits<double>::min();
    for(auto it=hkls.begin();it!=hkls.end();++it){
      int32_t h = get<0>(*it), k = get<1>(*it), l = get<2>(*it);
      const matrix<double,3,1> hkl{static_cast<double>(h),
                                   static_cast<double>(k),
                                   static_cast<double>(l)};
      //cout << "##############################" << endl;
      //cout << trans(hkl);
      for (size_t j=0;j!=parm.crystl.size();++j)
        ms[j] = get<1>(parm.crystl[j]).R*hkl;
      double intensity = 0;
      double sumw = 0;
      /*
      for (size_t j=0;j!=parm.crystl.size();++j){
        const double& weight = get<0>(parm.crystl[j]);
        const crystl_summand& crystl  = get<1>(parm.crystl[j]);
        const matrix<double,3,1> kin  = average_source.wvn*average_source.win;
        wouts[j] = normalize(ms[j]+kin);
        const matrix<double,3,1> w    = wouts[j];
        const matrix<double,3,1> Dw   = w-average_source.win;
        const matrix<double,3,1> m1   = average_source.wvn*Dw-ms[j];
        const matrix<double,3,1> m2   = (w*trans(w))*m1;
        const matrix<double,3,3> S    = constant_covariance_matrix(
            Dw,ms[j],average_source,crystl);
        const matrix<double,3,3> iS   = inv(S);
        const double e = abs(0.5*trans(m2)*iS*m2);
        const double lorentz = weight;//(trans(w)*iS*w*sqrt(2*pi));
        intensity+=e>LOG_DBL_MAX?0:lorentz*exp(-e);
        sumw+=weight;
      }
      intensity/=sumw;
      // intensity/=max_intensity;
      // partiality with average source and constant covariance matrix
      cout << "conservative estimate of intensity: " << endl;
      cout << intensity << " " << fraction << " " << max_intensity << endl;
      if ( intensity < pow(fraction,2) ) continue;
      intensity = 0;
      sumw = 0;
      */
      for (size_t i=0;i!=parm.source.size();++i)
      for (size_t j=0;j!=parm.crystl.size();++j){
        constexpr bool o = false;
        if (o) cout << "-------------" << endl;
        const double& source_weight  = get<0>(parm.source[i]);
        const double& crystl_weight  = get<0>(parm.crystl[j]);
        const double weight          = source_weight*crystl_weight;
        const source_summand& source = get<1>(parm.source[i]);
        const crystl_summand& crystl = get<1>(parm.crystl[j]);
        const double& wvn            = source.wvn;
        const matrix<double,3,1> kin = source.win*wvn;
        matrix<double,3,1>& w        = wouts[i*parm.crystl.size()+j];
        const matrix<double,3,1>& win= source.win;
        w = normalize(ms[j]+kin);
        if (o) cout << trans(w);
        //cout << trans(wouts[i*parm.crystl.size()+j]);
        //cout << exp(-0.5*trans(source.wvn*wouts[i]-ms[j])*inv(covariance_matrix(
        //  wouts[i],ms[j],source,crystl))*(source.wvn*wouts[i]-ms[j])) << endl;
        //for (size_t n=0;n!=2;++n)
        for (size_t n=0;n!=6;++n){
          const matrix<double,3,1> m1 = w-ms[j]/wvn-win;
          const matrix<double,3,3> S    =
            covariance_matrix(w-win,ms[j],source,crystl);
          const matrix<double,3,3> iS   = inv(S);
          const double e = pow(wvn,2u)*trans(m1)*iS*m1; 
          if (o) cout << e << endl;
          bool success = true;
          const matrix<double,3,1> tmp =
            optimize_wout(w,ms[j],source,crystl,success);
          w = tmp;
          if (!success) break;
          if (o) cout << trans(w);
        }
        if (o) cout << trans(ms[j]/source.wvn+source.win);
        //cout << trans(wouts[i*parm.crystl.size()+j]) << endl;
        //cout << exp(-0.5*trans(source.wvn*wouts[i]-ms[j])*inv(covariance_matrix(
        //  wouts[i],ms[j],source,crystl))*(source.wvn*wouts[i]-ms[j])) << endl;
        const matrix<double,3,1> m1 = w-ms[j]/wvn-win;
        const matrix<double,3,1> mp = (w*trans(w))*m1;
        //cout << trans(m1) << trans(mp);
        const matrix<double,3,3> S    =
          covariance_matrix(w-win,ms[j],source,crystl);
        const matrix<double,3,3> iS   = inv(S);
        if (o) cout << trans(m1);
        const double e = pow(wvn,2)*abs(0.5*trans(mp)*iS*(mp));
        if (o) cout << e << endl;
        //cout << e << endl;
        const double lorentz = weight/sqrt(2*pi*trans(w)*iS*w);
        //cout << lorentz << endl;
        intensity+=e>LOG_DBL_MAX?0:lorentz*exp(-e);
        sumw += weight;
      }
      intensity/=sumw;
      intensity/=max_intensity;
      //cout << "---------------------------" << endl;
      //cout << intensity << " " << fraction << " " << max_intensity << endl;
      if ( intensity < fraction ) continue;
      intensity = 0;
      unordered_map<const geometry::panel*,vector<prediction_ellipse>>
          subellipse_map;
      for (size_t i=0;i!=parm.source.size();++i)
      for (size_t j=0;j!=parm.crystl.size();++j){
        //cout << "---------------" << endl;
        const double& source_weight  = get<0>(parm.source[i]);
        const double& crystl_weight  = get<0>(parm.crystl[j]);
        const double  weight         = source_weight*crystl_weight/sumw;
        const source_summand& source = get<1>(parm.source[i]);
        const crystl_summand& crystl = get<1>(parm.crystl[j]);
        const double& wvn            = source.wvn;
        const matrix<double,3,1>& w  = wouts[i*parm.crystl.size()+j];
        const matrix<double,3,1>& win= source.win;
        const matrix<double,3,3> S   =
          covariance_matrix(w-win,ms[j],source,crystl);
        const matrix<double,3,3> iS = inv(S);
/* map k1 to the detector whos coordinate system is given by:     *
 *                                                                *
 *  / fs2x  ss2x  \  / fs \   / corner_x \  =  / x \              *
 *  | fs2y  ss2y  |  \ ss / + | corner_y |  =  | y |              *
 *  \ fs2z  ss2z  /           \ corner_z /  =  \ z /              *
 *                                                                *
 *   *  matrix D *   * x *     * offset *      * y *              */
        matrix<double,3,1> y;
        auto mapping = geom.map_to_fsss(y=w);
        matrix<double,2,1> x = get<0>(mapping);
        const panel* p = get<1>(mapping);
        if ( p == nullptr ) continue; // peak not on panel
        //cout << "finding projection" << endl;
        const double norm_y = length(y);
        const matrix<double,3,2> P = p->D/norm_y-y*trans(y)*p->D/pow(norm_y,3u);
        //cout << P << endl;
        const matrix<double,3,1> m0= wvn*(P*x+win-y/norm_y)+ms[j];
        //cout << m0 << endl;
        matrix<double,2,2> iS1 = trans(P)*iS*P;
        //cout << iS1 << endl;
        matrix<double,2,2>  S1 = inv(iS1);
        //cout << S1 << endl;
        //const matrix<double,2,1>  m1 = S1*trans(P)*iS*m0;
        // TODO c2 is fucked :(
        //const double c2 = abs(trans(m0)*iS*m0-trans(m1)*iS1*m1);
        const matrix<double,3,1>  mp = (w*trans(w))*(w-ms[j]/wvn-win);
        //cout << trans(w-ms[j]/wvn-win) << trans(mp);
        const double c2 = pow(wvn,2)*trans(mp)*iS*mp;
        //const double c2 = pow(wvn,2u)*
        //  abs(trans(w-ms[j]/wvn-win)*iS*(w-ms[j]/wvn-win));
        const matrix<double,2,2> _S1=S1
          +matrix<double,2,2>{1.0/oversampling,0,0,1.0/oversampling};
        const matrix<double,2,2> _iS1=inv(_S1);
        //cout << 0.5*c2 << endl;
        //cout << trans(x) << endl;
        //cout << trans(m1) << S1; // TODO m1 is fuckd
        const double lorentz = weight/sqrt(trans(w)*iS*w*2.0*pi);
        //cout << lorentz << endl;
        subellipse_map[p].push_back({x,_S1,_iS1,c2,lorentz});
        //cout << trans(w)*S*w << " "
        //     << frac
        //     << endl;
        intensity += lorentz*exp(-0.5*c2);
      }
      //cout << "final intensity = " << intensity/max_intensity << endl;
      if (subellipse_map.size()==0) continue;
      if (intensity>max_intensity) max_intensity=intensity; 
      vector<tuple<const geometry::panel*,
        vector<prediction_ellipse>>> subellipses;
      for (auto it=subellipse_map.begin();it!=subellipse_map.end();++it)
        subellipses.push_back({it->first,it->second});
      ellipses.push_back({reduce_encode(h,k,l,1),subellipses});
    }
    vector<tuple<uint64_t,vector<tuple<
      const geometry::panel*,
      vector<prediction_ellipse>
    >>>> filtered_ellipses;
    for (auto it0=ellipses.begin();it0!=ellipses.end();++it0){
      const uint64_t& index = get<0>(*it0);
      vector<tuple<const geometry::panel*,vector<prediction_ellipse>>> tmp0;
      double intensity = 0;
      for (auto it1=get<1>(*it0).begin();it1!=get<1>(*it0).end();++it1){
        const geometry::panel* p = get<0>(*it1);
        vector<prediction_ellipse> tmp1;
        for (auto it2=get<1>(*it1).begin();it2!=get<1>(*it1).end();++it2){
          tmp1.push_back(
            {it2->m,it2->S,it2->iS,it2->a2,it2->lorentz/max_intensity});
          intensity += tmp1.back().frac();
        }
        tmp0.push_back({p,tmp1});
      }
      cout << intensity << " " << fraction << endl;
      if (intensity < fraction) continue;
      filtered_ellipses.push_back({index,tmp0});
    }
    delete[] wouts;
    delete[] ms;
    return filtered_ellipses;
  }
  
  bool inline read_parameters(ifstream& file,parameters& parm){
    while (!file.eof()){
      if (file.peek()=='<'){
        //cout << "source line" << endl;
        file.ignore(1);
        tuple<double,source_summand> source;
        if (!(file>>get<0>(source)))        return false;
        if (!(file>>get<1>(source).win(0))) return false;
        if (!(file>>get<1>(source).win(1))) return false;
        if (!(file>>get<1>(source).win(2))) return false;
        if (!(file>>get<1>(source).wvn))    return false;
        if (!(file>>get<1>(source).bnd))    return false;
        double div;
        if (!(file>>div))                   return false;
        get<1>(source).div = pow(div,2u)*(identity_matrix<double>(3)
                            -get<1>(source).win*trans(get<1>(source).win));
        parm.source.push_back(source);
        file.ignore(numeric_limits<streamsize>::max(),'\n');
        continue;
      }
      if (file.peek()=='>'){
        //cout << "crystal line" << endl;
        file.ignore(1);
        tuple<double,crystl_summand> crystl;
        if (!(file>>get<0>(crystl)))        return false;
        if (!(file>>get<1>(crystl).R(0,0))) return false;
        if (!(file>>get<1>(crystl).R(0,1))) return false; 
        if (!(file>>get<1>(crystl).R(0,2))) return false; 
        if (!(file>>get<1>(crystl).R(1,0))) return false; 
        if (!(file>>get<1>(crystl).R(1,1))) return false; 
        if (!(file>>get<1>(crystl).R(1,2))) return false; 
        if (!(file>>get<1>(crystl).R(2,0))) return false; 
        if (!(file>>get<1>(crystl).R(2,1))) return false; 
        if (!(file>>get<1>(crystl).R(2,2))) return false; 
        get<1>(crystl).U = inv(get<1>(crystl).R);
        if (!(file>>get<1>(crystl).mosaicity)) return false;
        double peak;
        if (!(file>>peak)) break;
        get<1>(crystl).peak = pow(peak,2u)*identity_matrix<double>(3);
        if (!(file>>get<1>(crystl).strain)) return false;
        parm.crystl.push_back(crystl);
        file.ignore(numeric_limits<streamsize>::max(),'\n');
        continue;
      }
      file.ignore(1);
    }
    return !file.bad();
  }
}
#endif // PARTIALITY_H
