#ifndef PARTIALITY_H
#define PARTIALITY_H
#include "wmath.hpp"
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
  using std::conditional;
  using std::cout;
  using std::endl;
  using std::fill;
  using std::remainder;
  using std::fixed;
  using std::floor;
  using std::function;
  using std::get;
  using std::getline;
  using std::ifstream;
  using std::invoke_result;
  using std::isnan;
  using std::istream;
  using std::map;
  using std::max_element;
  using std::min;
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
  using wmath::circadd;
  using wmath::clip;
  using wmath::digits;
  using wmath::hash_functor;
  using wmath::inverf;
  using wmath::log2;
  using wmath::mean_variance;
  using wmath::mean_variance_mean_dw;
  using wmath::mean_variance_mean_dx;
  using wmath::mean_variance_M2_dx_ini;
  using wmath::mean_variance_M2_dx_acc;
  using wmath::mean_variance_M2_dw_ini;
  using wmath::mean_variance_M2_dw_acc;
  using wmath::mean_variance_sumw_dw;
  using wmath::mean_variance_sumw_dx;
  using wmath::popcount;
  using wmath::pow;
  using wmath::rol;
  
  typedef tuple<int32_t,int32_t,int32_t> IDX;

  /* Dk covariance contribution due to divergence
   * This models the incoming wave vector direction distribution.
   * It leads to a broadening of the observed peaks similar but discernable
   * from the reciprocal peak width.
   */
  matrix<double,3,3> inline S_divergence(
      const matrix<double,3,1>& win, // normed ingoing wave vector
      const double& div,
      const double& wvn){
    return pow(div*wvn,2u)*(identity_matrix<double>(3)-win*trans(win));
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
      const matrix<double,3,1>& x,
      const double& mosaicity
      ){
    const double m2 = mosaicity*mosaicity;
    const double nx2 = length_squared(x);
    return m2*(nx2*identity_matrix<double>(3)-x*trans(x));
  }

  matrix<double,3,1> inline cross_product(
      const matrix<double,3,1>& a,
      const matrix<double,3,1>& b){
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
      ){
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
      ){
    return (strain*x)*trans(strain*x);
  }
  
  struct source{
    matrix<double,3,1> win;  // incoming wave vector
    double wvn;              // wavenumber, reiprocal wavelength
    double bnd;              // bandwidth, in wavenumbers
    matrix<double,3,3> div;  // divergence
  };

  struct crystl{
    matrix<double,3,3> U;    // unit cell
    matrix<double,3,3> R;    // reciprocal unit cell { inv(U) }
    double mosaicity;        // mosaicity
    matrix<double,3,3> peak; // reciprocal peak
    double strain=0;         // crystal strain
    double a=1,b=0;          // scaling parameters a exp( - 0.5 b q² )
  };

  source const inline average_source(
      const vector<tuple<double,source>>& sources
      ) {
    double sumw = 0;
    matrix<double,3,1> mean_win{0,0,0};
    double mean_wvn = 0;
    matrix<double,3,3> M2_div{0,0,0,0,0,0,0,0,0};
    matrix<double,3,3> div{0,0,0,0,0,0,0,0,0};
    double M2_bnd = 0;
    double bnd = 0;
    for (auto it=sources.begin();it!=sources.end();++it){
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
    return source{mean_win,mean_wvn,bnd,div};
  }

  matrix<double,3,3> const inline constant_covariance_matrix(
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const source& source,
      const crystl& crystl
      ){
    return pow(source.wvn,2u)*source.div
          +pow(source.bnd,2u)*identity_matrix<double>(3) // isotropic
          +P_mosaicity(m,crystl.mosaicity)
          +P_strain(m,crystl.strain)
          +crystl.peak;
  }
  
  // covariance contribution of source
  matrix<double,3,3> const inline matrix_S(
      const matrix<double,3,1>& Dw,
      const source& source
      ){
    return pow(source.wvn,2u)*source.div
          +S_bandwidth(Dw,source.bnd);
  }
  
  // covariance contriubtion of crystal
  matrix<double,3,3> const inline matrix_P(
      const matrix<double,3,1>& m,
      const crystl& crystl
      ){
    return P_mosaicity(m,crystl.mosaicity)
          +P_strain(m,crystl.strain)
          +crystl.peak;
  }
  
  matrix<double,3,3> const inline covariance_matrix( // Dk covariance
      const matrix<double,3,1>& Dw,
      const matrix<double,3,1>& m,
      const struct source& source,
      const struct crystl& crystl
      ){
    /*return   pow(source.wvn,2u)*source.div
            +S_bandwidth(normalize(Dw),source.bnd)  // anisotropic
            +P_mosaicity(m,crystl.mosaicity)
            +P_strain(m,crystl.strain)
            +crystl.peak;*/
    return  matrix_S(Dw,source)+matrix_P(m,crystl);
  }

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

  matrix<double,3,3> const rotation_matrix(
      const double a,
      const matrix<double,3,1>& u
      ){
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

  struct rotation_target{
    const struct source& source;
    const struct crystl& crystl;
    const matrix<double,3,1>& wout;
    const matrix<double,3,1>& hkl;
    const matrix<double,3,1>& axis;
    const double& a;
    const double& c;
    const double operator()(
        const matrix<double,3,1>& x
        ) const {
      const matrix<double,3,1> w = normalize(matrix<double,3,1>
          {wout(0)+x(0),wout(1)+x(1),wout(2)});
      const double b = x(2);
      struct crystl crystl_rot = crystl;
      const matrix<double,3,3> Ma = rotation_matrix(a,axis);
      const matrix<double,3,3> Mb = rotation_matrix(b,axis);
      const matrix<double,3,3> Mc = rotation_matrix(c,axis);
      crystl_rot.U=Mb*crystl.U;
      crystl_rot.R=Mb*crystl.R;
      const matrix<double,3,1> ma = Ma*crystl.R    *hkl;
      const matrix<double,3,1> mb =    crystl_rot.R*hkl;
      const matrix<double,3,1> mc = Mc*crystl.R    *hkl;
      const matrix<double,3,1> Dw = w-source.win;
      const matrix<double,3,1> Dk = source.wvn*Dw;
      const matrix<double,3,3>  S = covariance_matrix(Dw,mb,source,crystl_rot);
      const matrix<double,3,3> iS = inv(S);
      const matrix<double,3,1>  d = Dk-mb;
      const double             v0 = trans(w)*S*w;
      const matrix<double,3,1> rv = cross_product(axis,mb);
      const double             v1 = v0*trans(w)*rv;
      return exp(-0.5*pow(double(tmp(trans(d)*w)),2)/v0)
        *abs((erf((c-b)/sqrt(2*v1))-erf((a-b)/sqrt(2*v1)))/2);
    }
  };
  
  // { (hkl) -> { position -> partiality } }
  vector<
    tuple
    <
      tuple<int32_t,int32_t,int32_t>, // {h,k,l}
      tuple<uint32_t,double>          // {pixel,flux}
    >
  >
  predict_rotation_image(
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const matrix<double,3,1>& axis,              // rotation axis
      const double a,                              // starting angle in rad
      const double b,                              // end angle in rad
      const double fraction = 1.0/64               // minimum partiality
      ){
    vector<
      tuple
      <
        tuple<int32_t,int32_t,int32_t>, // {h,k,l}
        tuple<uint32_t,double>          // {pixel,flux}
      >
    > prediction;
    const auto source = average_source(sources);
    const double max_e = -sqrt(2)*inverf(fraction);
    const double max_e2= pow(max_e,2);
    const matrix<double,3,1> kin = source.win*source.wvn;
    const matrix<double,3,3> Ra  = rotation_matrix(a,axis);
    const matrix<double,3,3> Rb  = rotation_matrix(b,axis);
    const double ab              = remainder((a+b)/2,2*pi);
    // half way between Ra and Rb
    const matrix<double,3,3> Rab2= rotation_matrix(ab,axis);
    const matrix<double,3,3> SR  = trans(crystl.R)*crystl.R;
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
      vector<IDX> todo;
      unordered_set<IDX,hash_functor<IDX>> done;
      // take one point on a panel
      const matrix<double,2,1> x{it->nfs/2.0,it->nss/2.0}; // the center
      // project it to the detector
      const matrix<double,3,1> y   = (*it)(x);
      // find closest hkl value
      const matrix<double,3,1> Dw  = normalize(y)-source.win;
      const matrix<double,3,1> hkl = crystl.U*Dw*source.wvn;
      const int32_t h = round(hkl(0)), k = round(hkl(1)), l = round(hkl(2));
      const IDX first_index({h,k,l});
      // add this reflection to todo list
      todo.push_back({h,k,l});
      while (todo.size()) { // while there is volume to grow into
        const IDX index = todo.back();
        todo.pop_back();
        // test each neighbouring reflection
        for (int32_t l=get<2>(index)-1;l<=get<2>(index)+1;++l)
        for (int32_t k=get<1>(index)-1;k<=get<1>(index)+1;++k)
        for (int32_t h=get<0>(index)-1;h<=get<0>(index)+1;++h) {
          // if this index has been examined before, skip ahead
          if (done.count (index)) continue;
          // else mark off that this index has been examined
          done.insert(index);
          const matrix<double,3,1> hkl{static_cast<double>(h),
                                       static_cast<double>(k),
                                       static_cast<double>(l)};
                matrix<double,3,1> m     = Rab2*crystl.R*hkl;
                matrix<double,3,1> wout  = normalize(m+kin);
                matrix<double,3,1> Dw    = wout-source.win;
                matrix<double,3,1> Dk    = source.wvn*Dw;
          const matrix<double,3,3> S1    =
            max_e*covariance_matrix(Dw,m,source,crystl)
                 +P_rotation(m,fmod(a-b,pi),axis)
                 +SR;
          // test if there is a reflection within
          // / h +- 0.5 \
          // | k +- 0.5 |
          // \ l +- 0.5 /
          // that wolud be excited
          const double e1 = trans(m-Dk)*inv(S1)*(m-Dk);
          if ( e1 > 1 ) continue;
          cout << h << " " << k << " " << l << endl;
#ifdef COMMENTED
          matrix<double,3,1> y = wout;
          matrix<double,2,1> fsss = (*it)(y);
          bool wasvalid = it->isvalid(fsss);
          rotation_target target{source,crystl,wout,hkl,axis,a,b};
          matrix<double,3,1> x{0.0,0.0,ab};
          dlib::find_max_using_approximate_derivatives(
              dlib::bfgs_search_strategy,
              dlib::objective_delta_stop_strategy(1e-20,1u<<4),
              target,
              x,
              numeric_limits<double>::max(),
              1e-12
              );
          wout(0)+= x(0);
          wout(1)+= x(1);
          wout = normalize(wout);
          fsss = (*it)(y=wout);
          // add the miller index to to todo list if it produces a position
          // on the detector or at least did produce a position on the detector
          // before optimization in case the optimization went off the deep end
          if (!it->isvalid(fsss)) {
            if (wasvalid) todo.push_back(index);
            continue;
          }
          todo.push_back(index);
          // find optimal rotation angle
          double partiality = target(x);
          if ( 2*partiality < fraction ) continue;
          m = rotation_matrix(x(2),axis)*crystl.R*hkl;
          partiality = 0;
          matrix<double,3,1>* wouts = new matrix<double,3,1>[sources.size()];
          double* max_as            = new double[sources.size()];
          for (size_t i=0;i!=sources.size();++i) {
            const auto& source = get<1>(sources[i]);
            rotation_target target{source,crystl,wout,hkl,axis,a,b};
            const matrix<double,3,1> _x = x;
            dlib::find_max_using_approximate_derivatives(
                dlib::bfgs_search_strategy,
                dlib::objective_delta_stop_strategy(1e-20,1u<<4),
                target,
                _x,
                numeric_limits<double>::max(),
                1e-12
                );
            partiality+= target(x);
            wout[i](0) = _x(0);
            wout[i](1) = _x(1);
            wout[i](2) = 1   ;
            wout[i] = normalize(wouts[i]);
            max_as[i]  = _x(2);
          }
          if ( partiality < fraction ) continue;
          // render shape
          const double o = oversampling;
          const double dpsfv = pow(o,-2); // +1;
          unordered_map<uint32_t,double> shape;
          matrix<double,3,1>  y           = wout;
          matrix<double,2,1>  fsss        = (*it)(y);
          const double norm_y             = length(y);
          const matrix<double,3,3> S      =
            max_e*(
                covariance_matrix(Dw,m,source,crystl)
                P_rotation(m,fmod(a-b,pi),axis));
          const matrix<double,3,2> P      =
            it->D/norm_y-y*trans(y)*it->D/pow(norm_y,3u);
                matrix<double,2,2>  iAS1  = trans(P)*inv(S)*P;
          const matrix<double,2,2>   AS1  =
            inv(iAS1)+dpsfv*identity_matrix<double>(2);
                                    iAS1  = inv(AS1);
          const matrix<double,2,2>  _AS1  = max_e2*AS1; 
          const matrix<double,2,2> _iAS1  = inv(_AS1);
          for (size_t i=0;i!=sources.size();++i) {
            const matrix<double,3,1>    m =
              rotation_matrix(max_as[i],axis)*crystl.R*hkl;
            const auto& source = get<1>(sources[i]);
            const matrix<double,3,1>  kin = source.win*source.wvn;
            const matrix<double,3,3>    S =
              covariance_matrix(Dw,m,source,crystl);
            const matrix<double,3,3>   _S =
              covariance_matrix(Dw,m,source,crystl)
              P_rotation(m,fmod(a-b,pi),axis);
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
            const matrix<double,2,1>   m0 = fsss;
            const matrix<double,2,2>  iS1 = trans(P)*inv(_S)*P;
            const matrix<double,2,2>   S1 = inv(iS1);
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
            for (size_t ss=min_ss;ss<=max_ss;++ss)
            for (size_t fs=min_fs;fs<=max_fs;++fs){
              const matrix<double,2,1> x{fs+0.5,ss+0.5};
              const size_t p = (*it)(fs,ss);
              if (trans(x-m0)*iAS1*(x-m0)+e0>max_e2) continue;
              if (trans(x-m0)*_iS1*(x-m0)>1) continue;
              for (size_t oss=0;oss!=oversampling;++oss)
              for (size_t ofs=0;ofs!=oversampling;++ofs){
                const matrix<double,2,1> x
                  {fs+(2*ofs+1)/(2*o),ss+(2*oss+1)/(2*o)};
                const matrix<double,3,1> y    = (*it)(x);
                const matrix<double,2,1> x0{fs+ofs/o,ss+oss/o};
                const matrix<double,2,1> x1{fs+(ofs+1)/o,ss+oss/o};
                const matrix<double,2,1> x2{fs+ofs/o,ss+(oss+1)/o};
                const matrix<double,2,1> ov{o,o};
                const double angular_coverage =
                  solid_angle((*it)(x0),(*it)(x1),(*it)(x2));
                const double        norm_y    = length(y);
                const matrix<double,3,2>    P =
                  it->D/norm_y-y*trans(y)*it->D/pow(norm_y,3u);
                const matrix<double,3,1> wout = y/norm_y;
                const matrix<double,3,3>    I =
                  pow(source.wvn,2)*P*o*trans(P*o);
                const matrix<double,3,1>   Dw = wout-source.win;
                const matrix<double,3,1>    k = source.wvn*wout;
                const matrix<double,3,1>   Dk = source.wvn*Dw;
                const matrix<double,3,1> diff = m-Dk;
                const matrix<double,3,3>    S =
                  covariance_matrix(Dw,m,source,crystl)+I;
                const double e = 0.5*trans(diff)*iS0*diff;
                const double v = trans(wout)*S0*wout;
                shape[(*it)(fs,ss)]+=angular_coverage*partiality;
              }
            }
      }
      vector<tuple<uint32_t,double>> unrolledshape;
      for (auto it=shape.begin();it!=shape.end();++it){
        unrolledshape.push_back(
            {
            it->first,
            get<0>(it->second),
            get<1>(it->second),
            get<2>(it->second)
            });
        }
        prediction.push_back(std::make_tuple(index,unrolledshape));
        }
      }
#endif // COMMENTED
      //delete[] wouts;
      }
    }
  }
  return prediction;
}

  enum class predict_mode : size_t {
    candidate_hkls,
    pixel_partiality,
    index_partiality
  };

  template<
    predict_mode mode,
    bool compute_derivatives
  >
  auto const inline get_predict_return_type(){
    if constexpr (mode==predict_mode::candidate_hkls) {
      return unordered_set<IDX,hash_functor<IDX>>{};
    } else {
      if constexpr (mode==predict_mode::pixel_partiality) {
        if constexpr (compute_derivatives){
          return vector<tuple<IDX,vector<
            tuple<uint32_t,double,double,double,crystl,crystl,crystl>>>>{};
        } else {
          return vector<tuple<IDX,vector<
            tuple<uint32_t,double,double,double>>>>{};
        }
      } else {
        if constexpr (mode==predict_mode::index_partiality) {
          if constexpr (compute_derivatives) {
            return vector<tuple<IDX,
                   double,crystl,
                   double,crystl,
                   double,crystl>>{};
          } else { // { index, flx, wvn, bnd, dfs, dss
            return vector<tuple<IDX,double,double,double,double,double>>{};
          }
        } else {
          return 0;
        }
      }
    }
  }

  template<predict_mode mode=predict_mode::candidate_hkls,
           bool compute_derivatives=false,
           size_t verbosity=0,
           size_t  oversampling=1>
  auto const inline predict( // one template to rule them all
      const vector<tuple<double,source>>& sources, // source
      const struct crystl& crystl,                 // crystal
      const geometry::geometry& geom,              // geometry
      const double fraction = 1.0/64               // minimum partiality
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
    const auto source = average_source(sources);
    const double max_e = -sqrt(2)*inverf(fraction);
    const double max_e2= pow(max_e,2);
    if constexpr (verbosity>3){
      cerr << "source.win =" << endl;
      cerr << source.win << endl;
      cerr << "crystl.R =" << endl;
      cerr << crystl.R << endl;
      cerr << "max_e = " << max_e << endl;
    }
    const matrix<double,3,1> kin = source.win*source.wvn;
    if constexpr (verbosity>3){
      cerr << "crystl.R = " << endl << crystl.R << endl;
      cerr << "max_e = " << max_e << endl;
      cerr << "kin = " << endl << kin << endl;
    }
    unordered_set<IDX,hash_functor<IDX>> hkls;
    auto prediction = get_predict_return_type<mode,compute_derivatives>();
    //matrix<double,3,3> SR{0,0,0,0,0,0,0,0,0,0};
    //const matrix<double,3,3> SR = crystl.R*trans(crystl.R);
    const matrix<double,3,3> SR = trans(crystl.R)*crystl.R;
    const double n0 = det(SR);
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
      file.read(reinterpret_cast<char*>(&crystl.mosaicity),8);
      file.read(reinterpret_cast<char*>(&peak            ),8);
      file.read(reinterpret_cast<char*>(&crystl.strain   ),8);
      crystl.U = inv(crystl.R);
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
}
#endif // PARTIALITY_H
