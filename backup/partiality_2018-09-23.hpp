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
  
  typedef tuple<int32_t,int32_t,int32_t> IDX;

  struct prediction_ellipse{
    const matrix<double,2,1> m;   // center of ellipse
    const matrix<double,2,2> S;   // covariance of cut
    const matrix<double,2,2> iS;  // inverse covariance of cut 
    const matrix<double,3,3> _S;  // covariance matrix
    const matrix<double,3,3> _iS; // inverse covariance matrix
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
  
  struct source{
    matrix<double,3,1> win;
    double wvn; // wavenumber, reiprocal wavelength
    double bnd; // bandwidth, in wavenumbers
    matrix<double,3,3> div;
  };

  struct crystl{
    matrix<double,3,3> U;    // unit cell
    matrix<double,3,3> R;    // reciprocal unit cell { inv(U) }
    double mosaicity;        // mosaicity
    matrix<double,3,3> peak; // reciprocal peak
    double strain;           // crystal strain
    double a,b;              // scaling parameters a exp( - 0.5 b q² )
  };

  struct parameters{
    vector<tuple<double,struct source>> sources;
    struct crystl crystl;
    source_summand const inline average_source() const {
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
      return source_summand{mean_win,mean_wvn,bnd,div};
    }
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
      const struct source& source,
      const struct crystl& crystl
      ){
    return   pow(source.wvn,2u)*source.div
            +S_bandwidth(normalize(Dw),source.bnd)  // anisotropic
            +P_mosaicity(m,crystl.mosaicity)
            +P_strain(m,crystl.strain)
            +crystl.peak;
    return  matrix_S(Dw,source)+matrix_P(m,crystl);
  }

  matrix<double,3,1> const inline optimize_wout(
      const matrix<double,3,1>& w,
      const matrix<double,3,1>& m,
      const struct source& source,
      const struct crystl& crystl,
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

  unordered_set<IDX,hash_functor<IDX>> const inline candidate_hkls(
      const struct source& source,
      const struct crystl& crystl,
      const geometry::geometry& geom,
      const double fraction
      ){
    const double max_e = -sqrt(2)*inverf(fraction);
    const matrix<double,3,1> kin = source.win*source.wvn;
    unordered_map<IDX,hash_functor<IDX>> hkls;
    vector<IDX> todo;
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
    // take one point on a panel
      const matrix<double,2,1> x{it->nfs/2.0,it->nss/2.0};
    // project to detector
      const matrix<double,3,1> y   = (*it)(x);
      const matrix<double,3,1> Dw  = normalize(y)-source.win;
      const matrix<double,3,1> hkl = crystl.U*Dw*source.wvn;
    // find closest hkl value
      int32_t h = round(hkl(0)), k = round(hkl(1)), l= round(hkl(2));
      todo.push_back({h,k,l});
    // add to todo
    while (todo.size()) for (auto it=todo.size();todo.end();++it){
       const IDX& index = todo.back();
       todo.pop_back();
       hkls.insert(IDX);
       for (int32_t l=get<2>(index)-1;l<=get<2>(index)+1;++l)
       for (int32_t k=get<1>(index)-1;k<=get<1>(index)+1;++k)
       for (int32_t h=get<0>(index)-1;h<=get<0>(index)+1;++h){
         const IDX index{h,k,l};
         if (hkls.count(index)) continue;
         const matrix<double,3,1> hkl{static_cast<double>(h),
                                      static_cast<double>(k),
                                      static_cast<double>(l)};
         const matrix<double,3,1> m    = crystl.R*hkl;
         const matrix<double,3,1> wout = normalize(m+kin);
         const matrix<double,3,1> Dw   = wout-source.win;
         const matrix<double,3,1> x    = crystl.U*Dw*source.wvn;
         const matrix<double,3,3> S    =
           covariance_matrix(
              Dw,
              m,
              source,
              crystl);
         // TODO think wether to calculate difference on w or k
         const double e = (m-Dw)*inv(S)*(m-Dw);
         if ( ( length(hkl-x) > 1 ) && ( e > max_e ) ) continue;
         hkls.push_back(index);
       }
    }
    return hkls;
  }

  unordered_set<IDX,hash_functor<IDX>> const inline candidate_hkls_old(
      const struct source& source,
      const struct crystl& crystl,
      const geometry::geometry& geom,
      const double fraction
      ){
    unordered_set<IDX,hash_functor<IDX>> hkls;
    const double max_e = -sqrt(2)*inverf(fraction);
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
      for (size_t ss=0;ss<=it->nss;++ss) for (size_t fs=0;fs<=it->nfs;++fs){
        const matrix<double,2,1> x{double(fs+0.5),double(ss+0.5)};
        const matrix<double,3,1> y = (*it)(x);
        const matrix<double,3,1> Dw= normalize(y)-source.win;
        const matrix<double,3,1> m = crystl.U*Dw*source.wvn;
        const matrix<double,3,3> S =
          covariance_matrix(
              Dw,
              m,
              source,
              crystl);
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
    return hkls;
  }

  enum predict_mode{
    candidate_hkls,
    pixel_partiality,
    index_partiality
  };
  
  template<
    predict_mode mode,
    bool compute_derivatives
  >
  auto const inline get_predict_return_type(){
    if constexpr (mode==candidate_hkls)                      // candidate_hkls
      return unordered_set<IDX,hash_functor<IDX>>;
    if constexpr (mode==predict_mode::pixel_partiality)      // pixel_partiality
      if constexpr (compute_derivatives)
        return vector<tuple<IDX,vector<
          tuple<uint32_t,double,crystl,double,crystl,double,crystl>>>>;
      else
        return vector<tuple<IDX,vector<tuple<uint32_t,double,double,double>>>>;
    if constexpr (mode==predict_mode::index_partiality)      // index_partiality
      if constexpr (compute_derivatives)
        return vector<tuple<IDX,vector<tuple<double,cryst>>>>;
      else
        return vector<tuple<IDX,vector<double>>>;
    else return 0;
  }

  template<predict_mode mode=predict_mode::pixel_partiality,
           bool compute_derivatives=false,
           bool test_derivatives=false,
           bool test_projection=false,
           size_t verbosity=0,
           size_t oversampling=1>
  auto const inline predict( // one template to rule them all
      const struct parameters& parm,   // parameters
      const geometry::geometry& geom,  // geometry
      const double fraction = 1.0/64,  // minimum partiality
      ){
    const auto source = parm.average_source();
    const double max_e = -sqrt(2)*inverf(fraction);
    const matrix<double,3,1> kin = source.win*source.wvn;
    unordered_map<IDX,hash_functor<IDX>> hkls;
    auto prediction = get_predict_return_type<mode>();
    vector<IDX> todo;
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it){
    // take one point on a panel
      const matrix<double,2,1> x{it->nfs/2.0,it->nss/2.0};
    // project to detector
      const matrix<double,3,1> y   = (*it)(x);
      const matrix<double,3,1> Dw  = normalize(y)-source.win;
      const matrix<double,3,1> hkl = crystl.U*Dw*source.wvn;
    // find closest hkl value
      int32_t h = round(hkl(0)), k = round(hkl(1)), l= round(hkl(2));
      todo.push_back({h,k,l});
    // add to todo
    while (todo.size()) for (auto it=todo.size();todo.end();++it){
      const IDX& index = todo.back();
      todo.pop_back();
      hkls.insert(IDX);
      for (int32_t l=get<2>(index)-1;l<=get<2>(index)+1;++l)
      for (int32_t k=get<1>(index)-1;k<=get<1>(index)+1;++k)
      for (int32_t h=get<0>(index)-1;h<=get<0>(index)+1;++h){
        const IDX index{h,k,l};
        if (hkls.count(index)) continue;
        const matrix<double,3,1> hkl{static_cast<double>(h),
                                     static_cast<double>(k),
                                     static_cast<double>(l)};
        const matrix<double,3,1> m    = crystl.R*hkl;
        const matrix<double,3,1> wout = normalize(m+kin);
        const matrix<double,3,1> Dw   = wout-source.win;
        const matrix<double,3,1> x    = crystl.U*Dw*source.wvn;
        const matrix<double,3,3> S    =
          covariance_matrix(
             Dw,
             m,
             source,
             crystl);
        // TODO think wether to calculate difference on w or k
        const double e = (m-Dw)*inv(S)*(m-Dw);
        if ( ( length(hkl-x) > 1 ) && ( e > max_e ) ) continue;
        todo.push_back(index);
        if constexpr(mode==prediction_mode::candidate_hkls) {continue;} else{
        const matrix<double,3,1> wouts =
          new matrix<double,3,1>[parm.sources.size()];
        double p = 0;
        for (size_t it=parm.sources.begin();it!=parm.sources.end();++it){
          const auto& source = *it;
          auto& w = wouts[i];
          w = normalize(m+kin);
          for (size_t j=0;j!=6;++j){
            bool success = true;
            const matrix<double,3,1> tmp =
            optimize_wout(w,m,source_component,crystl,success);
            if (!success) break;
            w = tmp;
          }
        }
        if constexpr(mode==prediction_mode::index_partiality){
        prediciton.push_back{index,p};
        continue;
        } else if constexpr(
            (mode==prediction_mode::pixel_partiality)
          ||(mode==prediction_mode::pixel_partiality_dcryst) {
        vector<size_t>                      pxls;
        vector<double>                      flxs;
        vector<tuple<double,double,double>> wvns;
        vector<double>                      bnds;
        vector<crystl>               flxs_dcryst;
        vector<crystl>               wvns_dcryst;
        vector<crystl>               bnds_dcryst;
        //project average peakshape to compute:
        // reciprocal peakshape
        const matrix<double,3,3> S = ?; // TODO
        matrix<double,2,2> _S = ?; // TODO
        matrix<double,2,2> _iS = inv(_S);
        const double ext_fs = max_e*sqrt(_S(0,0));
        const double ext_ss = max_e*sqrt(_S(1,1));
        //then go over pixels that lie within contour levels of S
        for (size_t i=0;
                    i!=(mode==prediction_mode::pixel_partiality_dcryst);
                  ++i){
        for (size_t it=parm.sources.begin();it!=parm.sources.end();++it){
          // TODO oversampling
          const size_t min_fs = clip(size_t(round(m(0)-ext_fs)),0,p->nfs-1);
          const size_t max_fs = clip(size_t(round(m(0)+ext_fs)),0,p->nfs-1);
          const size_t min_ss = clip(size_t(round(m(1)-ext_ss)),0,p->nss-1);
          const size_t max_ss = clip(size_t(round(m(1)+ext_ss)),0,p->nss-1);
          for (size_t ss=min_ss;ss<=max_ss;++ss)
          for (size_t fs=min_fs;fs<=max_fs;++fs){
            // TODO accumulate flux, wvn, bnd and if *_dcryst accumulate
            // their derivatives wrt cryst as well
            if (i==1){
              // TODO calculate derivatives
            }
          }
          }
        }
        }
        prediction.resize(prediction.size()+1);
        for (size_t i=0;i!=pxls.size();++i){
          const size_t& pxl = pxls[i];
          const double& flx = flxs[i];
          const double& wvn = get<1>(wvns[i]);
          const double& bnd = bnds[i];
          if constexpr(mode==prediction_mode::pixel_partiality){
            prediction.back().push_back{pixel,flux,wvn,bnd};
          }else if constexpr(mode==prediction_mode::pixel_partiality_dcryst){
            prediction.back().push_back({
               pixel,
               flx,flxs_dcryst[i],
               wvn,wvns_dcryst[i],
               bnd,bnds_dcryst[i]
               });
          }
        }
        delete[] wouts;
      }
      }
    }
    }
    if constexpr (mode==predict_mode::candidate_hkls) return hkls;
    else return prediction;
  }

  template<size_t mode>

  unordered_set<IDX,hash_functor<IDX>> const inline candidate_hkls(
      const struct parameters& parm,
      const geometry::geometry& geom,
      const double fraction
      ){
    return candidate_hkls(parm.average_source(),parm.crystl,geom,fraction);
  }
  
  vector<tuple<IDX,vector<tuple<
    const geometry::panel*,
    vector<prediction_ellipse>
  >>>>
  const inline get_prediction_ellipses(
    const struct source& source,
    const struct crystl& crystl,
    const geometry::geometry& geom,
    const double fraction = 1.0/64,
    const size_t oversampling = 1
    ){
    vector<tuple<tuple<int32_t,int32_t,int32_t>,vector<tuple<
      const geometry::panel*,
      vector<prediction_ellipse>
    >>>> ellipses;
    const auto hkls = candidate_hkls(source,crystl,geom,fraction);
    matrix<double,3,1>* wouts= new matrix<double,3,1>[source.components.size()];
    double max_intensity     = numeric_limits<double>::min();
    for(auto it=hkls.begin();it!=hkls.end();++it){
      int32_t h = get<0>(*it), k = get<1>(*it), l = get<2>(*it);
      const matrix<double,3,1> hkl{static_cast<double>(h),
                                   static_cast<double>(k),
                                   static_cast<double>(l)};
      const matrix<double,3,1> m = crystl.R*hkl;
      double intensity = 0;
      double sumw = 0;
      for (size_t i=0;i!=source.components.size();++i){
        constexpr bool o = false;
        if (o) cout << "-------------" << endl;
        const double& weight         = get<0>(source.components[i]);
        const struct source_summand& source_summand
                                     = get<1>(source.components[i]);
        const double& wvn            = source_summand.wvn;
        const matrix<double,3,1> kin = source_summand.win*wvn;
        matrix<double,3,1>& w        = wouts[i];
        const matrix<double,3,1>& win= source_summand.win;
        w = normalize(m+kin);
        if (o) cout << trans(w);
        for (size_t n=0;n!=6;++n){
          const matrix<double,3,1> m1   = w-m/wvn-win;
          const matrix<double,3,3> S    =
            covariance_matrix(w-win,m,source_summand,crystl);
          const matrix<double,3,3> iS   = inv(S);
          const double e = pow(wvn,2u)*trans(m1)*iS*m1; 
          if (o) cout << e << endl;
          bool success = true;
          const matrix<double,3,1> tmp =
            optimize_wout(w,m,source_summand,crystl,success);
          if (!success) break;
          w = tmp;
          if (o) cout << trans(w);
        }
        if (o) cout << trans(ms[j]/source.wvn+source.win);
        const matrix<double,3,1> m1 = w-m/wvn-win;
        const matrix<double,3,1> mp = m1;
        const matrix<double,3,3> S    =
          covariance_matrix(w-win,m,source_summand,crystl);
        const matrix<double,3,3> iS   = inv(S);
        if (o) cout << trans(m1);
        const double e = pow(wvn,2)*abs(0.5*trans(mp)*iS*(mp));
        if (o) cout << e << endl;
        const double lorentz = weight/sqrt(2*pi*trans(w)*S*w);
        intensity+=e>LOG_DBL_MAX?0:lorentz*exp(-e);
        sumw += lorentz; // weight;
      }
      intensity/=sumw;
      //cout << "---------------------------" << endl;
      //cout << intensity << " " << fraction << " " << max_intensity << endl;
      if ( intensity/max_intensity < fraction ) continue;
      intensity = 0;
      sumw = 0;
      unordered_map<const geometry::panel*,vector<prediction_ellipse>>
          subellipse_map;
      for (size_t i=0;i!=source.components.size();++i){
        const double& weight         = get<0>(source.components[i]);
        const struct source_summand& source_summand
                                     = get<1>(source.components[i]);
        const double& wvn            = source.wvn;
        const matrix<double,3,1>& w  = wouts[i];
        const matrix<double,3,1>& win= source.win;
        const matrix<double,3,3>  S  =
          covariance_matrix(w-win,m,source_summand,crystl);
        const matrix<double,3,3> iS  = inv(S);
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
        const double norm_y = length(y);
        const matrix<double,3,2> P = p->D/norm_y-y*trans(y)*p->D/pow(norm_y,3u);
        const matrix<double,3,1> m0= wvn*(P*x+win-y/norm_y)+ms[j];
        matrix<double,2,2> iS1 = trans(P)*iS*P;
        matrix<double,2,2>  S1 = inv(iS1);
        //const matrix<double,2,1>  m1 = S1*trans(P)*iS*m0;
        // TODO c2 is fucked :(
        //const double c2 = abs(trans(m0)*iS*m0-trans(m1)*iS1*m1);
        //const matrix<double,3,1>  mp = (w*trans(w))*(w-ms[j]/wvn-win);
        const matrix<double,3,1>  mp = (w-ms[j]/wvn-win);
        const double c2 = pow(wvn,2)*trans(mp)*iS*mp;
        //const double c2 = pow(wvn,2u)*
        //  abs(trans(w-ms[j]/wvn-win)*iS*(w-ms[j]/wvn-win));
        const matrix<double,2,2> _S1=S1
          +matrix<double,2,2>{1.0/oversampling,0,0,1.0/oversampling};
        const matrix<double,2,2> _iS1=inv(_S1);
        // TODO m1 is fuckd
        const double lorentz = weight/sqrt(trans(w)*S*w*2.0*pi);
        subellipse_map[p].push_back({x,_S1,_iS1,S,iS,c2,lorentz});
        intensity += lorentz*exp(-0.5*c2);
        sumw += lorentz; // weight;
      }
      intensity/=sumw;
      if (subellipse_map.size()==0) continue;
      if (intensity>max_intensity) max_intensity=intensity; 
      vector<tuple<const geometry::panel*,
        vector<prediction_ellipse>>> subellipses;
      for (auto it=subellipse_map.begin();it!=subellipse_map.end();++it)
        subellipses.push_back({it->first,it->second});
      ellipses.push_back({{h,k,l},subellipses});
    }
  // TODO: is this really necessary?
    vector<tuple<tuple<int32_t,int32_t,int32_t>,vector<tuple<
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
            {it2->m,it2->S,it2->iS,it2->a2,it2->lorentz});
          intensity += tmp1.back().frac();
        }
        tmp0.push_back({p,tmp1});
      }
      //cout << intensity << " " << fraction << endl;
      if (intensity < fraction) continue;
      filtered_ellipses.push_back({index,tmp0});
    }
    delete[] wouts;
    return filtered_ellipses;
  }
  
  /* TODO prediction_entry_diff
  struct prediction_entry_diff{
    // when using poisson distribution λ = flux*wvn
    // when using gamma distribution then parameter
    // ϑ = flux*adu_per_ev*adu_per_ev*(wvn+bnd)/wvn
    // k = flux*adu_per_ev*(wvn+bnd)
    // but gamma distribution is not sufficient
    // it should be convoluted with gaussian(0,adu_per_ev)
    private:
      double sumw   = 0;
      double M2     = 0;
      double M3     = 0;
    public:
      double flx    = 0; // flux
      double flx_da = 0;
      double flx_db = 0;
      double flx_dm = 0;
      double flx_dp = 0;
      double flx_ds = 0;
      matrix<double,3,3> flx_dR = zeros_matrix<double,3,3>();
      double wvn    = 0; // wavenumber
      double wvn_da = 0;
      double wvn_db = 0;
      double wvn_dm = 0;
      double wvn_dp = 0;
      double wvn_ds = 0;
      matrix<double,3,3> wvn_dR = zeros_matrix<double,3,3>();
      void add_wvn_bnd(const double& x,const double& w,const double& v){
        mean_variance(x,c,sumw,wvn,M2);
        M3+=v;
      }
      double const bnd() const {
        return M2+M3;
        // diff = 2*w*(1-w/sumw)*(x-mean)
      }
      double const poisson_lambda() const {
        return flx*wvn;
      }
  };
  */
  
  struct prediction_entry{
    // when using poisson distribution λ = flux*wvn
    // when using gamma distribution then parameter
    // ϑ = flux*adu_per_ev*adu_per_ev*(wvn+bnd)/wvn
    // k = flux*adu_per_ev*(wvn+bnd)
    // but gamma distribution is not sufficient
    // it should be convoluted with gaussian(0,adu_per_ev)
    private:
      double sumw   = 0;
      double M2     = 0;
      double M3     = 0;
    public:
      double flx    = 0; // flux
      matrix<double,3,3> flx_dR = zeros_matrix<double,3,3>();
      double wvn    = 0; // wavenumber
      matrix<double,3,3> wvn_dR = zeros_matrix<double,3,3>();
      void add_wvn_bnd(const double& x,const double& w,const double& v){
        mean_variance(x,c,sumw,wvn,M2);
        M3+=v;
      }
      double const bnd() const {
        return M2+M3;
      }
      double const poisson_lambda(const double& i) const {
        return i*flx*wvn;
      }
  };
  
  // TODO: derivatives
  // TODO: rewrite to
  // vector<tuple<IDX,vector<tuple<uint32_t,prediction_entry_diff>>>>
  // TODO: add wrapper for predict(parm,geom,frac,o)
  unordered_multimap<
    size_t,
    tuple<
      tuple<int32_t,int32_t,int32_t>,
      prediction_entry // prediction_entry_diff
    >
  > const inline predict(
      const parameters&         parm,
      const geometry::geometry& geom,
    const double&                  a,
    const double&                  b,
    const double&               frac,
    const size_t&                  o
      ){
    const auto average_source = source.average_source();
    const auto hkls = candidate_hkls(average_source,crystl,geom,fraction);
    const matrix<double,3,1> kin = average_source.win*average_source.wvn;
    matrix<double,3,1>* wouts= new matrix<double,3,1>[source.components.size()];
    matrix<double,3,3>* S0s  = new matrix<double,3,3>[source.components.size()];
    for(auto it=hkls.begin();it!=hkls.end();++it){
      const matrix<double,3,1> hkl{static_cast<double>(get<0>(*it)),
                                   static_cast<double>(get<1>(*it)),
                                   static_cast<double>(get<2>(*it)}; 
      const matrix<double,3,1> m = crystl.R*hkl;
      //test average_source for partiaity, if partiality less than frac continue;
      matrix<double,3,1> w = normalize(m+kin);
      for (size_t j=0;j!=6;++j){
        bool success = true;
        const matrix<double,3,1> tmp =
          optimize_wout(w,m,average_source,crystl,success);
        if (!success) break;
        w = tmp;
      }
      const double e = ;
      //project average peakshape to compute:
      // reciprocal peakshape
      const matrix<double,3,3> S1 =
            +P_mosaicity(m,crystl.mosaicity)
            +P_strain(m,crystl.strain)
            +crystl.peak;
      matrix<double,3,3> _S;
      matrix<double,3,3> _iS = inv(_S);
      then go over pixels that lie within contour levels of S
      const double ext_fs = max_e*sqrt(_S(0,0));
      const double ext_ss = max_e*sqrt(_S(1,1));
      const size_t min_fs = clip(size_t(round(m(0)-ext_fs)),0,p->nfs-1);
      const size_t max_fs = clip(size_t(round(m(0)+ext_fs)),0,p->nfs-1);
      const size_t min_ss = clip(size_t(round(m(1)-ext_ss)),0,p->nss-1);
      const size_t max_ss = clip(size_t(round(m(1)+ext_ss)),0,p->nss-1);
      // compute S, iS, and wout for each source component
      for (size_t i=0;i!=source.components.size();++i){
        const auto& source_component = source.components[i];
        auto& w = wouts[i];
        w = normalize(m+kin);
        for (size_t j=0;j!=6;++j){
          bool success = true;
          const matrix<double,3,1> tmp =
            optimize_wout(w,m,source_component,crystl,success);
          if (!success) break;
          w = tmp;
        }
        auto& S0 = S0s[i];
        S0 = pow(source.wvn,2u)*source.div;
      }
      for (size_t ss=min_ss;ss<=max_ss;++ss)
      for (size_t fs=min_fs;fs<=max_fs;++fs){
        const matrix<double,2,1> x{fs+0.5,ss+0.5};
        const matrix<double,2,1> d = x-m;
        if (trans(d)*_iS*d > max_e*max_e) continue;
        // S0 = source
        // S1 = crystl
        // S2 = S0+S1
        // S3 = inv(inv(S0)+inv(S1))
        // add a small component to S depending on the position on the detector
        // to compensate for finite sampling to enshure σ > 1pixel/sampling 
        prediction_entry entry;
        for (size_t i=0;i!=o;++i){
          const double dfs = fs+(i+0.5)/o;
          const double dss = ss+(i+0.5)/o;
          for (source components){
            // wavelength and variance on cut along Dw :
            const double ibnd0 = trans(Dw)*iS*Dw;
            const double wvn0  = bnd*trans(m)*iS*Dw;
            const double ibnd  = 1.0/source_component.bnd + ibnd0;
            const double wvn   =
              ibnd*(source_component.wvn/source_component.bnd + wvn0/ibnd0);
            const double e     = 0.5*(
                trans(wvn*Dw-m)*iS*(wvn*Dw-m)
               +);
            const double flux  =
               exp(-0.5*trans(wvn*Dw-m)*iS*(wvn*Dw-m));                    
            const double c     = exp(-0.5*trans(wvn*Dw-m)*iS*(wvn*Dw-m));
            entry.add_bnd_wvn(wvn,c,1.0/ibnd);
          }
        }
        prediction.insert({pos,{it->first,prediction_entry}});
      }
    }
    delete[] wouts;
    delete[] S0s;
    return result;
  }
  
  parameters const inline deserialize_parameters(ifstream& file)
    struct parameters parm;
    uint32_t n;
    file.read(reinterpret_cast<char*>(&n),4);
    parm.sources.resize(n); 
    for (i=0;i!=n;++i){
      struct source& source = parm.sources[i];
      file.read(reinterpret_cast<char*>(&get<0>(source       )),8);
      file.read(reinterpret_cast<char*>(&get<1>(source.win(0))),8);
      file.read(reinterpret_cast<char*>(&get<1>(source.win(1))),8);
      file.read(reinterpret_cast<char*>(&get<1>(source.win(2))),8);
      file.read(reinterpret_cast<char*>(&get<1>(source.wvn   )),8);
      file.read(reinterpret_cast<char*>(&get<1>(source.bnd   )),8);
      double div;
      file.read(reinterpret_cast<char*>(&div),8);
      get<1>(source).div = pow(div,2u)*(identity_matrix<double>(3)
                          -get<1>(source).win*trans(get<1>(source).win));
    }
    struct crystl& crystl;
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
    return parm;
  }
  /*
  bool inline read_parameters(
      ifstream& file,
      struct source& source,
      struct crystl& crystl){
    while (!file.eof()){
      if (file.peek()=='<'){
        //cout << "source line" << endl;
        file.ignore(1);
        tuple<double,struct source_summand> source_summand;
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
      if (file.peek()=='#'){
        return !file.bad();
      }
      file.ignore(1);
    }
    return !file.bad();
  }
  bool read_parameterstream(ifstream& file,vector<parameters>& parms){
    // TODO
    while (true){
      parameters parm;
      if (read_parameters(file,parm)) parms.push_back(parm);
      else return false;
      if (file.eof()) return true;
    }
  }
  */
}
#endif // PARTIALITY_H
