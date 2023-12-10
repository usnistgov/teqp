#pragma once

#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include <optional>
#include <utility>
#include <set>
#include <unordered_set>

#include "teqp/math/pow_templates.hpp"
#include "Eigen/Dense"

namespace teqp{

namespace GERG2004{

const std::vector<std::string> component_names = {"methane", "nitrogen","carbondioxide","ethane","propane","n-butane","isobutane","n-pentane","isopentane","n-hexane","n-heptane","n-octane","hydrogen", "oxygen","carbonmonoxide","water","helium","argon"};

struct PureInfo{
    double rhoc_molm3, Tc_K, M_kgmol;
};

/// Get the pure fluid information for a fluid from GERG-2004 monograph
PureInfo get_pure_info(const std::string& name){
    
    // From Table A3.5 from GERG 2004 monograph
    // Data are in mol/dm3, K, kg/mol
    static std::map<std::string, PureInfo> data_map = {
        {"methane", {10.139342719,190.564000000,16.042460}},
        {"nitrogen", {11.183900000,126.192000000,28.013400}},
        {"carbondioxide", {10.624978698,304.128200000,44.009500}},
        {"ethane", {6.870854540,305.322000000,30.069040}},
        {"propane", {5.000043088,369.825000000,44.095620}},
        {"n-butane", {3.920016792,425.125000000,58.122200}},
        {"isobutane", {3.860142940,407.817000000,58.122200}},
        {"n-pentane", {3.215577588,469.700000000,72.148780}},
        {"isopentane", {3.271018581,460.350000000,72.148780}},
        {"n-hexane", {2.705877875,507.820000000,86.175360}},
        {"n-heptane", {2.315324434,540.130000000,100.201940}},
        {"n-octane", {2.056404127,569.320000000,114.228520}},
        {"hydrogen", {14.940000000,33.190000000 ,2.015880}},
        {"oxygen", {13.630000000,154.595000000,31.998800}},
        {"carbonmonoxide", {10.850000000,132.800000000,28.010100}},
        {"water", {17.873716090,647.096000000,18.015280}},
        {"helium", {17.399000000,5.195300000,4.002602}},
        {"argon", {13.407429659,150.687000000,39.948000}}
    };
    auto data = data_map.at(name);
    data.rhoc_molm3 *= 1000; // mol/dm^3 -> mol/m^3
    data.M_kgmol /= 1000; // kg/kmol -> kg/mol
    return data;
}

struct PureCoeffs{
    std::vector<double> n, t, d, c, l;
    std::set<std::size_t> sizes(){ return {n.size(), t.size(), d.size(), c.size(), l.size()}; }
};

PureCoeffs get_pure_coeffs(const std::string& fluid){
    
    // From Table A3.2 from GERG 2004 monograph
    static std::map<std::string, std::vector<double>> n_dict_mne = {
        {"methane", {0.57335704239162,-0.16760687523730e1,0.23405291834916,-0.21947376343441,0.16369201404128e-1,0.15004406389280e-1,0.98990489492918e-1,0.58382770929055,-0.74786867560390,0.30033302857974,0.20985543806568,-0.18590151133061e-1,-0.15782558339049,0.12716735220791,-0.32019743894346e-1,-0.68049729364536e-1,0.24291412853736e-1,0.51440451639444e-2,-0.19084949733532e-1,0.55229677241291e-2,-0.44197392976085e-2,0.40061416708429e-1,-0.33752085907575e-1,-0.25127658213357e-2}},
        {"nitrogen", { 0.59889711801201,-0.16941557480731e1,0.24579736191718,-0.23722456755175,0.17954918715141e-1,0.14592875720215e-1,0.10008065936206,0.73157115385532,-0.88372272336366,0.31887660246708,0.20766491728799,-0.19379315454158e-1,-0.16936641554983,0.13546846041701,-0.33066712095307e-1,-0.60690817018557e-1,0.12797548292871e-1,0.58743664107299e-2,-0.18451951971969e-1,0.47226622042472e-2,-0.52024079680599e-2,0.43563505956635e-1,-0.36251690750939e-1,-0.28974026866543e-2}},
        {"ethane", { 0.63596780450714,-0.17377981785459e1,0.28914060926272,-0.33714276845694,0.22405964699561e-1,0.15715424886913e-1,0.11450634253745,0.10612049379745e1,-0.12855224439423e1,0.39414630777652,0.31390924682041,-0.21592277117247e-1,-0.21723666564905,-0.28999574439489,0.42321173025732,0.46434100259260e-1,-0.13138398329741,0.11492850364368e-1,-0.33387688429909e-1,0.15183171583644e-1,-0.47610805647657e-2,0.46917166277885e-1,-0.39401755804649e-1,-0.32569956247611e-2}}
    };
    
    static std::map<std::string, std::vector<double>> n_dict_main = {
        {"propane", {0.10403973107358e1,-0.28318404081403e1,0.84393809606294,-0.76559591850023e-1,0.94697373057280e-1,0.24796475497006e-3,0.27743760422870,-0.43846000648377e-1,-0.26991064784350,-0.69313413089860e-1,-0.29632145981653e-1,0.14040126751380e-1}},
        {"n-butane", { 0.10626277411455e1,-0.28620951828350e1,0.88738233403777,-0.12570581155345,0.10286308708106,0.25358040602654e-3,0.32325200233982,-0.37950761057432e-1,-0.32534802014452,-0.79050969051011e-1,-0.20636720547775e-1,0.57053809334750e-2}},
        {"isobutane", {0.10429331589100e1,-0.28184272548892e1,0.86176232397850,-0.10613619452487,0.98615749302134e-1,0.23948208682322e-3,0.30330004856950,-0.41598156135099e-1,-0.29991937470058,-0.80369342764109e-1,-0.29761373251151e-1,0.13059630303140e-1}},
        {"n-pentane", {0.10968643098001e1,-0.29988888298061e1,0.99516886799212,-0.16170708558539,0.11334460072775,0.26760595150748e-3,0.40979881986931,-0.40876423083075e-1,-0.38169482469447,-0.10931956843993,-0.32073223327990e-1,0.16877016216975e-1}},
        {"isopentane", {0.11017531966644e1,-0.30082368531980e1,0.99411904271336,-0.14008636562629,0.11193995351286,0.29548042541230e-3,0.36370108598133,-0.48236083488293e-1,-0.35100280270615,-0.10185043812047,-0.35242601785454e-1,0.19756797599888e-1}},
        {"n-hexane", {0.10553238013661e1,-0.26120615890629e1,0.76613882967260,-0.29770320622459,0.11879907733358,0.27922861062617e-3,0.46347589844105,0.11433196980297e-1,-0.48256968738131,-0.93750558924659e-1,-0.67273247155994e-2,-0.51141583585428e-2}},
        {"n-heptane", {0.10543747645262e1,-0.26500681506144e1,0.81730047827543,-0.30451391253428,0.12253868710800,0.27266472743928e-3,0.49865825681670,-0.71432815084176e-3,-0.54236895525450,-0.13801821610756,-0.61595287380011e-2,0.48602510393022e-3}},
        {"n-octane", {0.10722544875633e1,-0.24632951172003e1,0.65386674054928,-0.36324974085628,0.12713269626764,0.30713572777930e-3,0.52656856987540,0.19362862857653e-1,-0.58939426849155,-0.14069963991934,-0.78966330500036e-2,0.33036597968109e-2}},
        {"oxygen", { 0.88878286369701,-0.24879433312148e1,0.59750190775886,0.96501817061881e-2,0.71970428712770e-1,0.22337443000195e-3,0.18558686391474,-0.38129368035760e-1,-0.15352245383006,-0.26726814910919e-1,-0.25675298677127e-1,0.95714302123668e-2}},
        {"carbonmonoxide", {0.92310041400851,-0.24885845205800e1,0.58095213783396,0.28859164394654e-1,0.70256257276544e-1,0.21687043269488e-3,0.13758331015182,-0.51501116343466e-1,-0.14865357483379,-0.38857100886810e-1,-0.29100433948943e-1,0.14155684466279e-1}},
        {"argon", {0.85095714803969,-0.24003222943480e1,0.54127841476466,0.16919770692538e-1,0.68825965019035e-1,0.21428032815338e-3,0.17429895321992,-0.33654495604194e-1,-0.13526799857691,-0.16387350791552e-1,-0.24987666851475e-1,0.88769204815709e-2}}
    };
    
    if (n_dict_main.find(fluid) != n_dict_main.end()){
        PureCoeffs pc;
        pc.n = n_dict_main[fluid],
        pc.t = {0.250,1.125,1.500,1.375,0.250,0.875,0.625,1.750,3.625,3.625,14.500,12.000};
        pc.d = {1,1,1,2,3,7,2,5,1,4,3,4};
        pc.c = {0,0,0,0,0,0,1,1,1,1,1,1};
        pc.l = {0,0,0,0,0,0,1,1,2,2,3,3};
        return pc;
    }
    else if (n_dict_mne.find(fluid) != n_dict_mne.end()){
        PureCoeffs pc;
        pc.n = n_dict_mne.at(fluid);
        pc.t = {0.125,1.125,0.375,1.125,0.625,1.500,0.625,2.625,2.750,2.125,2.000,1.750,4.500,4.750,5.000,4.000,4.500,7.500,14.000,11.500,26.000,28.000,30.000,16.000};
        pc.d = {1,1,2,2,4,4,1,1,1,2,3,6,2,3,3,4,4,2,3,4,5,6,6,7};
        pc.c = {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        pc.l = {0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,6,6,6,6};
        return pc;
    }
    else if (fluid == "carbondioxide"){
        PureCoeffs pc;
        pc.n = {0.52646564804653,-0.14995725042592e1, 0.27329786733782, 0.12949500022786, 0.15404088341841,-0.58186950946814,-0.18022494838296,-0.95389904072812e-1,-0.80486819317679e-2,-0.35547751273090e-1,-0.28079014882405,-0.82435890081677e-1, 0.10832427979006e-1,-0.67073993161097e-2,-0.46827907600524e-2,-0.28359911832177e-1, 0.19500174744098e-1,-0.21609137507166, 0.43772794926972,-0.22130790113593, 0.15190189957331e-1,-0.15380948953300e-1};
        pc.t = {0.000,1.250,1.625,0.375,0.375,1.375,1.125,1.375,0.125,1.625,3.750,3.500,7.500,8.000,6.000,16.000,11.000,24.000,26.000,28.000,24.000,26.000};
        pc.d = {1,1,2,3,3,3,4,5,6,6,1,4,1,1,3,3,4,5,5,5,5,5};
        pc.c = {0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        pc.l = {0,0,0,0,1,1,1,1,1,1,2,2,3,3,3,3,3,5,5,5,6,6};
        return pc;
    }
    else if (fluid == "hydrogen"){
        PureCoeffs pc;
        pc.n = {0.53579928451252e1,-0.62050252530595e1, 0.13830241327086,-0.71397954896129e-1, 0.15474053959733e-1,-0.14976806405771,-0.26368723988451e-1, 0.56681303156066e-1,-0.60063958030436e-1,-0.45043942027132, 0.42478840244500,-0.21997640827139e-1,-0.10499521374530e-1,-0.28955902866816e-2};
        pc.t = {0.500,0.625,0.375,0.625,1.125,2.625,0.000,0.250,1.375,4.000,4.250,5.000,8.000,8.000};
        pc.d = {1,1,2,2,4,1,5,5,5,1,1,2,5,1};
        pc.c = {0,0,0,0,0,1,1,1,1,1,1,1,1,1};
        pc.l = {0,0,0,0,0,1,1,1,1,2,2,3,3,5};
        return pc;
    }
    else if (fluid == "water"){
        PureCoeffs pc;
        pc.n = {0.82728408749586,-0.18602220416584e1,-0.11199009613744e1,0.15635753976056,0.87375844859025,-0.36674403715731,0.53987893432436e-1,0.10957690214499e1,0.53213037828563e-1,0.13050533930825e-1,-0.41079520434476,0.14637443344120,-0.55726838623719e-1,-0.11201774143800e-1,-0.66062758068099e-2,0.46918522004538e-2};
        pc.t = {0.500,1.250,1.875,0.125,1.500,1.000,0.750,1.500,0.625,2.625,5.000,4.000,4.500,3.000,4.000,6.000};
        pc.d = {1,1,1,2,2,3,4,1,5,5,1,2,4,4,1,1};
        pc.c = {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};
        pc.l = {0,0,0,0,0,0,0,1,1,1,2,2,2,3,5,5};
        return pc;
    }
    else if (fluid == "helium"){
        PureCoeffs pc;
        pc.n = {-0.45579024006737,0.12516390754925e1,-0.15438231650621e1,0.20467489707221e-1,-0.34476212380781,-0.20858459512787e-1,0.16227414711778e-1,-0.57471818200892e-1,0.19462416430715e-1,-0.33295680123020e-1,-0.10863577372367e-1,-0.22173365245954e-1};
        pc.t = {0.000,0.125,0.750,1.000,0.750,2.625,0.125,1.250,2.000,1.000,4.500,5.000};
        pc.d = {1,1,1,4,1,3,5,5,5,2,1,2};
        pc.c = {0,0,0,0,1,1,1,1,1,1,1,1};
        pc.l = {0,0,0,0,1,1,1,1,1,2,3,3};
        return pc;
    }
    else{
        throw std::invalid_argument("unable to load pure coefficients for " + fluid);
    }
}

struct BetasGammas {
    double betaT, gammaT, betaV, gammaV;
};

BetasGammas get_betasgammas(const std::string&fluid1, const std::string &fluid2){
    
    // From Table A3.8 of GERG 2004 monograph
    std::map<std::pair<std::string, std::string>,BetasGammas> BIP_data = {
        {{"methane","nitrogen"}, {0.998721377,1.013950311,0.998098830,0.979273013}},
        {{"methane","carbondioxide"}, {0.999518072,1.002806594,1.022624490,0.975665369}},
        {{"methane","ethane"}, {0.997547866,1.006617867,0.996336508,1.049707697}},
        {{"methane","propane"}, {1.004827070,1.038470657,0.989680305,1.098655531}},
        {{"methane","n-butane"}, {0.979105972,1.045375122,0.994174910,1.171607691}},
        {{"methane","isobutane"}, {1.011240388,1.054319053,0.980315756,1.161117729}},
        {{"methane","n-pentane"}, {0.948330120,1.124508039,0.992127525,1.249173968}},
        {{"methane","isopentane"}, {1.000000000,1.343685343,1.000000000,1.188899743}},
        {{"methane","n-hexane"}, {0.958015294,1.052643846,0.981844797,1.330570181}},
        {{"methane","n-heptane"}, {0.962050831,1.156655935,0.977431529,1.379850328}},
        {{"methane","n-octane"}, {0.994740603,1.116549372,0.957473785,1.449245409}},
        {{"methane","hydrogen"}, {1.000000000,1.018702573,1.000000000,1.352643115}},
        {{"methane","oxygen"}, {1.000000000,1.000000000,1.000000000,0.950000000}},
        {{"methane","carbonmonoxide"}, {0.997340772,1.006102927,0.987411732,0.987473033}},
        {{"methane","water"}, {1.012783169,1.585018334,1.063333913,0.775810513}},
        {{"methane","helium"}, {1.000000000,0.881405683,1.000000000,3.159776855}},
        {{"methane","argon"}, {1.034630259,1.014678542,0.990954281,0.989843388}},
        {{"nitrogen","carbondioxide"}, {0.977794634,1.047578256,1.005894529,1.107654104}},
        {{"nitrogen","ethane"}, {0.978880168,1.042352891,1.007671428,1.098650964}},
        {{"nitrogen","propane"}, {0.974424681,1.081025408,1.002677329,1.201264026}},
        {{"nitrogen","n-butane"}, {0.996082610,1.146949309,0.994515234,1.304886838}},
        {{"nitrogen","isobutane"}, {0.986415830,1.100576129,0.992868130,1.284462634}},
        {{"nitrogen","n-pentane"}, {1.000000000,1.078877166,1.000000000,1.419029041}},
        {{"nitrogen","isopentane"}, {1.000000000,1.154135439,1.000000000,1.381770770}},
        {{"nitrogen","n-hexane"}, {1.000000000,1.195952177,1.000000000,1.472607971}},
        {{"nitrogen","n-heptane"}, {1.000000000,1.404554090,1.000000000,1.520975334}},
        {{"nitrogen","n-octane"}, {1.000000000,1.186067025,1.000000000,1.733280051}},
        {{"nitrogen","hydrogen"}, {0.972532065,0.970115357,0.946134337,1.175696583}},
        {{"nitrogen","oxygen"}, {0.999521770,0.997082328,0.997190589,0.995157044}},
        {{"nitrogen","carbonmonoxide"}, {1.000000000,1.008690943,1.000000000,0.993425388}},
        {{"nitrogen","water"}, {1.000000000,1.094749685,1.000000000,0.968808467}},
        {{"nitrogen","helium"}, {0.969501055,0.932629867,0.692868765,1.471831580}},
        {{"nitrogen","argon"}, {1.004166412,1.002212182,0.999069843,0.990034831}},
        {{"carbondioxide","ethane"}, {1.002525718,1.032876701,1.013871147,0.900949530}},
        {{"carbondioxide","propane"}, {0.996898004,1.047596298,1.033620538,0.908772477}},
        {{"carbondioxide","n-butane"}, {1.174760923,1.222437324,1.018171004,0.911498231}},
        {{"carbondioxide","isobutane"}, {1.076551882,1.081909003,1.023339824,0.929982936}},
        {{"carbondioxide","n-pentane"}, {1.024311498,1.068406078,1.027000795,0.979217302}},
        {{"carbondioxide","isopentane"}, {1.060793104,1.116793198,1.019180957,0.961218039}},
        {{"carbondioxide","n-hexane"}, {1.000000000,0.851343711,1.000000000,1.038675574}},
        {{"carbondioxide","n-heptane"}, {1.205469976,1.164585914,1.011806317,1.046169823}},
        {{"carbondioxide","n-octane"}, {1.026169373,1.104043935,1.029690780,1.074455386}},
        {{"carbondioxide","hydrogen"}, {0.904142159,1.152792550,0.942320195,1.782924792}},
        {{"carbondioxide","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"carbondioxide","carbonmonoxide"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"carbondioxide","water"}, {0.949055959,1.542328793,0.997372205,0.775453996}},
        {{"carbondioxide","helium"}, {0.846647561,0.864141549,0.768377630,3.207456948}},
        {{"carbondioxide","argon"}, {1.008392428,1.029205465,0.996512863,1.050971635}},
        {{"ethane","propane"}, {0.997607277,1.003034720,0.996199694,1.014730190}},
        {{"ethane","n-butane"}, {0.999157205,1.006179146,0.999130554,1.034832749}},
        {{"ethane","isobutane"}, {1.000000000,1.006616886,1.000000000,1.033283811}},
        {{"ethane","n-pentane"}, {0.993851009,1.026085655,0.998688946,1.066665676}},
        {{"ethane","isopentane"} , {1.000000000,1.045439246,1.000000000,1.021150247}},
        {{"ethane","n-hexane"}, {1.000000000,1.169701102,1.000000000,1.092177796}},
        {{"ethane","n-heptane"}, {1.000000000,1.057666085,1.000000000,1.134532014}},
        {{"ethane","n-octane"}, {1.007469726,1.071917985,0.984068272,1.168636194}},
        {{"ethane","hydrogen"}, {0.925367171,1.106072040,0.932969831,1.902008495}},
        {{"ethane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"ethane","carbonmonoxide"}, {1.000000000,1.201417898,1.000000000,1.069224728}},
        {{"ethane","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"ethane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"ethane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"propane","n-butane"}, {0.999795868,1.003264179,1.000310289,1.007392782}},
        {{"propane","isobutane"}, {0.999243146,1.001156119,0.998012298,1.005250774}},
        {{"propane","n-pentane"}, {1.044919431,1.019921513,0.996484021,1.008344412}},
        {{"propane","isopentane"}, {1.040459289,0.999432118,0.994364425,1.003269500}},
        {{"propane","n-hexane"}, {1.000000000,1.057872566,1.000000000,1.025657518}},
        {{"propane","n-heptane"}, {1.000000000,1.079648053,1.000000000,1.050044169}},
        {{"propane","n-octane"}, {1.000000000,1.102764612,1.000000000,1.063694129}},
        {{"propane","hydrogen"}, {1.000000000,1.074006110,1.000000000,2.308215191}},
        {{"propane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"propane","carbonmonoxide"}, {1.000000000,1.108143673,1.000000000,1.197564208}},
        {{"propane","water"}, {1.000000000,1.011759763,1.000000000,0.600340961}},
        {{"propane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"propane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-butane","isobutane"}, {1.000880464,1.000414440,1.000077547,1.001432824}},
        {{"n-butane","n-pentane"}, {1.000000000,1.018159650,1.000000000,1.002143640}},
        {{"n-butane","isopentane"}, {1.000000000,1.002728262,1.000000000,1.000792201}},
        {{"n-butane","n-hexane"}, {1.000000000,1.034995284,1.000000000,1.009157060}},
        {{"n-butane","n-heptane"}, {1.000000000,1.019174227,1.000000000,1.021283378}},
        {{"n-butane","n-octane"}, {1.000000000,1.046905515,1.000000000,1.033180106}},
        {{"n-butane","hydrogen"}, {1.000000000,1.232939523,1.000000000,2.509259945}},
        {{"n-butane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-butane","carbonmonoxide"}, {1.000000000,1.084740904,1.000000000,1.174055065}},
        {{"n-butane","water"}, {1.000000000,1.223638763,1.000000000,0.615512682}},
        {{"n-butane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-butane","argon"}, {1.000000000,1.214638734,1.000000000,1.245039498}},
        {{"isobutane","n-pentane"}, {1.000000000,1.002779804,1.000000000,1.002495889}},
        {{"isobutane","isopentane"}, {1.000000000,1.002284197,1.000000000,1.001835788}},
        {{"isobutane","n-hexane"}, {1.000000000,1.010493989,1.000000000,1.006018054}},
        {{"isobutane","n-heptane"}, {1.000000000,1.021668316,1.000000000,1.009885760}},
        {{"isobutane","n-octane"}, {1.000000000,1.032807063,1.000000000,1.013945424}},
        {{"isobutane","hydrogen"}, {1.000000000,1.147595688,1.000000000,1.895305393}},
        {{"isobutane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isobutane","carbonmonoxide"}, {1.000000000,1.087272232,1.000000000,1.161523504}},
        {{"isobutane","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isobutane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isobutane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-pentane","isopentane"}, {1.000000000,1.000024352,1.000000000,1.000050537}},
        {{"n-pentane","n-hexane"}, {1.000000000,1.002480637,1.000000000,1.000761237}},
        {{"n-pentane","n-heptane"}, {1.000000000,1.008972412,1.000000000,1.002441051}},
        {{"n-pentane","n-octane"}, {1.000000000,1.069223964,1.000000000,1.016422347}},
        {{"n-pentane","hydrogen"}, {1.000000000,1.188334783,1.000000000,2.013859174}},
        {{"n-pentane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-pentane","carbonmonoxide"}, {1.000000000,1.119954454,1.000000000,1.206195595}},
        {{"n-pentane","water"}, {1.000000000,0.956677310,1.000000000,0.447666011}},
        {{"n-pentane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-pentane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isopentane","n-hexane"}, {1.000000000,1.002996055,1.000000000,1.001204174}},
        {{"isopentane","n-heptane"}, {1.000000000,1.009928531,1.000000000,1.003194615}},
        {{"isopentane","n-octane"}, {1.000000000,1.017880981,1.000000000,1.005647480}},
        {{"isopentane","hydrogen"}, {1.000000000,1.184339122,1.000000000,1.996386669}},
        {{"isopentane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isopentane","carbonmonoxide"}, {1.000000000,1.116693501,1.000000000,1.199475627}},
        {{"isopentane","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isopentane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"isopentane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-hexane","n-heptane"}, {1.000000000,1.001508227,1.000000000,0.999762786}},
        {{"n-hexane","n-octane"}, {1.000000000,1.006268954,1.000000000,1.001633952}},
        {{"n-hexane","hydrogen"}, {1.000000000,1.243461678,1.000000000,3.021197546}},
        {{"n-hexane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-hexane","carbonmonoxide"}, {1.000000000,1.155145836,1.000000000,1.233435828}},
        {{"n-hexane","water"}, {1.000000000,1.170217596,1.000000000,0.569681333}},
        {{"n-hexane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-hexane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-heptane","n-octane"}, {1.000000000,1.006767176,1.000000000,0.998793111}},
        {{"n-heptane","hydrogen"}, {1.000000000,1.159131722,1.000000000,3.169143057}},
        {{"n-heptane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-heptane","carbonmonoxide"}, {1.000000000,1.190354273,1.000000000,1.256295219}},
        {{"n-heptane","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-heptane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-heptane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-octane","hydrogen"}, {1.000000000,1.305249405,1.000000000,2.191555216}},
        {{"n-octane","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-octane","carbonmonoxide"}, {1.000000000,1.219206702,1.000000000,1.276744779}},
        {{"n-octane","water"}, {1.000000000,0.599484191,1.000000000,0.662072469}},
        {{"n-octane","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"n-octane","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"hydrogen","oxygen"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"hydrogen","carbonmonoxide"}, {1.000000000,1.121416201,1.000000000,1.377504607}},
        {{"hydrogen","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"hydrogen","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"hydrogen","argon"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"oxygen","carbonmonoxide"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"oxygen","water"}, {1.000000000,1.143174289,1.000000000,0.964767932}},
        {{"oxygen","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"oxygen","argon"}, {0.999746847,0.993907223,1.000023103,0.990430423}},
        {{"carbonmonoxide","water"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"carbonmonoxide","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"carbonmonoxide","argon"}, {1.000000000,1.159720623,1.000000000,0.954215746}},
        {{"water","helium"}, {1.000000000,1.000000000,1.000000000,1.000000000}},
        {{"water","argon"}, {1.000000000,1.038993495,1.000000000,1.070941866}},
        {{"helium","argon"}, {1.00000000,1.00000000,1.00000000,1.00000000}}
    };
    
    try{
        // Try in the given order
        return BIP_data.at(std::make_pair(fluid1, fluid2));
    }
    catch(...){
        // Try the other order
        try{
            auto bg = BIP_data.at(std::make_pair(fluid2, fluid1));
            // Because the order is backwards, need to take the reciprocals of the beta values
            bg.betaT = 1/bg.betaT;
            bg.betaV = 1/bg.betaV;
            return bg;
        }
        catch(...){
            throw std::invalid_argument("Unable to obtain BIP for the pair {" + fluid1 + "," + fluid2 + "}");
        }
    }
}

std::optional<double> get_Fij(const std::string& fluid1, const std::string& fluid2, bool ok_missing=true){
    
    /// Table A3.6 from GERG 2004 monograph
    static std::map<std::pair<std::string, std::string>, double> Fij_dict = {
        {{"methane","nitrogen"}, 0.100000000000e1},
        {{"methane","carbondioxide"}, 0.100000000000e1},
        {{"methane","ethane"}, 0.100000000000e1},
        {{"methane","propane"}, 0.100000000000e1},
        {{"methane","n-butane"}, 0.100000000000e1},
        {{"methane","isobutane"}, 0.771035405688},
        {{"methane","hydrogen"}, 0.100000000000e1},
        {{"nitrogen","carbondioxide"}, 0.100000000000e1},
        {{"nitrogen","ethane"}, 0.100000000000e1},
        {{"ethane","propane"}, 0.130424765150},
        {{"ethane","n-butane"}, 0.281570073085},
        {{"ethane","isobutane"}, 0.260632376098},
        {{"propane","n-butane"}, 0.312572600489e-1},
        {{"propane","isobutane"}, -0.551609771024e-1},
        {{"n-butane","isobutane"}, -0.551240293009e-1}
    };
    try{
        // Try in the given order
        return Fij_dict.at(std::make_pair(fluid1, fluid2));
    }
    catch(...){
        // Try the other order
        try{
            return Fij_dict.at(std::make_pair(fluid2, fluid1));
        }
        catch(...){
            if (ok_missing){
                return std::nullopt;
            }
            else{
                throw std::invalid_argument("Unable to obtain Fij for the pair {" + fluid1 + "," + fluid2 + "}");
            }
        }
    }
}

struct DepartureCoeffs{
    std::vector<double> n, t, d, eta, beta, gamma, epsilon;
    std::set<std::size_t> sizes(){ return {n.size(), t.size(), d.size(), eta.size(), beta.size(), gamma.size(), epsilon.size()}; }
};

DepartureCoeffs get_departurecoeffs(const std::string&fluid1, const std::string &fluid2){
    
    std::pair<std::string, std::string> sortedpair = std::minmax(fluid1, fluid2);
    auto sortpair = [](const std::string &n1, const std::string& n2) ->std::pair<std::string, std::string> { return std::minmax(n1, n2); };
    
    // The set of pairnames using the generalized form
    const std::set<std::pair<std::string, std::string>> generalized = {
        sortpair("methane","n-butane"), sortpair("methane","isobutane"), sortpair("ethane","propane"),
        sortpair("ethane","n-butane"), sortpair("ethane","isobutane"), sortpair("propane","n-butane"),
        sortpair("propane","isobutane"), sortpair("n-butane","isobutane")
    };
    
    if (sortedpair == sortpair("methane","nitrogen")){
        DepartureCoeffs dc;
        dc.n = {-0.98038985517335e-2,0.42487270143005e-3,-0.34800214576142e-1 ,-0.13333813013896 ,-0.11993694974627e-1 ,0.69243379775168e-1 ,-0.31022508148249 ,0.24495491753226 ,0.22369816716981};
        dc.d = {1,4,1,2,2,2,2,2,3};
        dc.t = {0.000,1.850,7.850,5.400,0.000,0.750,2.800,4.450,4.250};
        dc.eta = {0,0,1.000,1.000,0.250,0.000,0.000,0.000,0.000};
        dc.epsilon = {0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
        dc.beta = {0,0,1.0,1.0,2.5,3.0,3.0,3.0,3.0};
        dc.gamma = {0,0,0.500,0.500,0.500,0.500,0.500,0.500,0.500};
        return dc;
    }
    if (sortedpair == sortpair("methane","carbondioxide")){
        DepartureCoeffs dc;
        dc.n = {-0.10859387354942,0.80228576727389e-1,-0.93303985115717e-2,0.40989274005848e-1,-0.24338019772494,0.23855347281124};
        dc.d = {1,2,3,1,2,3};
        dc.t = {2.600,1.950,0.000,3.950,7.950,8.000};
        dc.eta = {0,0,0,1.000,0.500,0.000};
        dc.epsilon = {0,0,0,0.5,0.5,0.5};
        dc.beta = {0,0,0,1.0,2.0,3.0};
        dc.gamma = {0,0,0,0.500,0.500,0.500};
        return dc;
    }
    if (sortedpair == sortpair("ethane", "methane")){
        DepartureCoeffs dc;
        dc.n = {-0.80926050298746e-3,-0.75381925080059e-3,-0.41618768891219e-1,-0.23452173681569,0.14003840584586,0.63281744807738e-1,-0.34660425848809e-1,-0.23918747334251,0.19855255066891e-2,0.61777746171555e1,-0.69575358271105e1,0.10630185306388e1};
        dc.t = {0.650,1.550,3.100,5.900,7.050,3.350,1.200,5.800,2.700,0.450,0.550,1.950};
        dc.d = {3,4,1,2,2,2,2,2,2,3,3,3};
        dc.eta = {0,0,1.000,1.000,1.000,0.875,0.750,0.500,0.000,0.000,0.000,0.000};
        dc.epsilon = {0,0,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500};
        dc.beta = {0,0,1.000,1.000,1.000,1.250,1.500,2.000,3.000,3.000,3.000,3.000};
        dc.gamma = {0,0,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500,0.500};
        return dc;
    }
    if (sortedpair == sortpair("propane", "methane")){
        DepartureCoeffs dc;
        dc.n = {0.13746429958576e-1,-0.74425012129552e-2,-0.45516600213685e-2,-0.54546603350237e-2, 0.23682016824471e-2, 0.18007763721438,-0.44773942932486, 0.19327374888200e-1,-0.30632197804624};
        dc.t = {1.850,3.950,0.000,1.850,3.850,5.250,3.850,0.200,6.500};
        dc.d = {3,3,4,4,4,1,1,1,2};
        dc.eta = {0,0,0,0,0,0.250,0.250,0.000,0.000};
        dc.epsilon = {0,0,0,0,0,0.500,0.500,0.500,0.500};
        dc.beta = {0,0,0,0,0,0.750,1.000,2.000,3.000};
        dc.gamma = {0,0,0,0,0,0.500,0.500,0.500,0.500};
        return dc;
    }
    if (sortedpair == sortpair("nitrogen", "carbondioxide")){
        DepartureCoeffs dc;
        dc.n = {0.28661625028399,-0.10919833861247,-0.11374032082270e1,0.76580544237358,0.42638000926819e2,0.17673538204534};
        dc.t = {1.850,1.400,3.200,2.500,8.000,3.750};
        dc.d = {2,3,1,1,1,2};
        dc.eta = {0,0,0.250,0.250,0.000,0.000};
        dc.epsilon = {0,0,0.500,0.500,0.500,0.500};
        dc.beta = {0,0,0.750,1.000,2.000,3.000};
        dc.gamma = {0,0,0.500,0.500,0.500,0.500};
        return dc;
    }
    if (sortedpair == sortpair("nitrogen", "ethane")){
        DepartureCoeffs dc;
        dc.n = {-0.47376518126608,0.48961193461001,-0.57011062090535e-2,-0.19966820041320,-0.69411103101723,0.69226192739021};
        dc.d = {2,2,3,1,2,2};
        dc.t = {0.000,0.050,0.000,3.650,4.900,4.450};
        dc.eta = {0,0,0,1.000,1.000,0.875};
        dc.epsilon = {0,0,0,0.500, 0.500, 0.500};
        dc.beta = {0,0,0,1.000,1.000,1.250};
        dc.gamma = {0,0,0,0.500,0.500,0.500};
        return dc;
    }
    else if (sortedpair == sortpair("methane", "hydrogen")){
        DepartureCoeffs dc;
        dc.n = {-0.25157134971934,-0.62203841111983e-2,0.88850315184396e-1,-0.35592212573239e-1};
        dc.t = {2.000,-1.000,1.750,1.400};
        dc.d = {1,3,3,4};
        dc.eta = {0,0,0,0};
        dc.epsilon = {0,0,0,0};
        dc.beta = {0,0,0,0};
        dc.gamma = {0,0,0,0};
        return dc;
    }
    else if (generalized.find(sortedpair) != generalized.end()){
        DepartureCoeffs dc;
        dc.n = {0.25574776844118e1,-0.79846357136353e1,0.47859131465806e1,-0.73265392369587,0.13805471345312e1,0.28349603476365,-0.49087385940425,-0.10291888921447,0.11836314681968,0.55527385721943e-4};
        dc.d = {1,1,1,2,2,3,3,4,4,4};
        dc.t = {1.000,1.550,1.700,0.250,1.350,0.000,1.250,0.000,0.700,5.400};
        dc.eta = {0,0,0,0,0,0,0,0,0,0};
        dc.epsilon = {0,0,0,0,0,0,0,0,0,0};
        dc.beta = {0,0,0,0,0,0,0,0,0,0};
        dc.gamma = {0,0,0,0,0,0,0,0,0,0};
        return dc;
    }
    else{
        throw std::invalid_argument("could not get departure coeffs for {" + fluid1 + "," + fluid2 + "}");
    }
}

// ***********************************************************
// ***********************************************************
//          Pures, Reducing, Corresponding States
// ***********************************************************
// ***********************************************************

/**
\f$ \alpha^{\rm r}=\displaystyle\sum_i n_i \delta^{d_i} \tau^{t_i} \exp(-c_i\delta^{l_i})\f$
*/
class GERG2004PureFluidEOS {
private:
    PureCoeffs pc;
    std::vector<int> l_i;
    auto get_li(std::vector<double>&el){
        std::vector<int> li(el.size());
        for (auto i = 0; i < el.size(); ++i){
            li[i] = static_cast<int>(el[i]);
        }
        return li;
    }
    
public:
    GERG2004PureFluidEOS(const std::string& name): pc(get_pure_coeffs(name)), l_i(get_li(pc.l)){}
    
    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        if (l_i.size() == 0 && pc.n.size() > 0) {
            throw std::invalid_argument("l_i cannot be zero length if some terms are provided");
        }
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < pc.n.size(); ++i) {
                r = r + pc.n[i] * exp(pc.t[i] * lntau - pc.c[i] * powi(delta, l_i[i])) * powi(delta, static_cast<int>(pc.d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < pc.n.size(); ++i) {
                r = r + pc.n[i] * exp(pc.t[i] * lntau + pc.d[i] * lndelta - pc.c[i] * powi(delta, l_i[i]));
            }
        }
        return forceeval(r);
    }
};

class GERG2004Reducing{
public:
    // As a structure to allow them to be initialized in one
    // pass through the fluids
    struct TcVc{ std::vector<double> Tc_K, vc_m3mol; };
    struct Matrices {Eigen::ArrayXXd betaT, gammaT, betaV, gammaV, YT, Yv; };
private:
    TcVc get_Tcvc(const std::vector<std::string>& names){
        std::vector<double> Tc(names.size()), vc(names.size());
        std::size_t i = 0;
        for (auto &name : names){
            auto pd = get_pure_info(name);
            Tc[i] = pd.Tc_K;
            vc[i] = 1.0/pd.rhoc_molm3;
            i++;
        }
        return TcVc{Tc, vc};
    }
    Matrices get_matrices(const std::vector<std::string>& names){
        Matrices m;
        std::size_t N = names.size();
        m.betaT.resize(N,N); m.gammaT.resize(N,N); m.betaV.resize(N,N); m.gammaV.resize(N,N); m.YT.resize(N,N);  m.Yv.resize(N,N);
        const auto& Tc = m_Tcvc.Tc_K;
        const auto& vc = m_Tcvc.vc_m3mol;
        for (auto i = 0; i < N; ++i){
            for (auto j = i+1; j < N; ++j){
                auto bg = get_betasgammas(names[i], names[j]);
                m.betaT(i,j) = bg.betaT; m.betaT(j,i) = 1/bg.betaT;
                m.gammaT(i,j) = bg.gammaT; m.gammaT(j,i) = bg.gammaT;
                m.betaV(i,j) = bg.betaV; m.betaV(j,i) = 1/bg.betaV;
                m.gammaV(i,j) = bg.gammaV; m.gammaV(j,i) = bg.gammaV;
                m.YT(i,j) = m.betaT(i,j)*m.gammaT(i,j)*sqrt(Tc[i]*Tc[j]);
                m.YT(j,i) = m.betaT(j,i)*m.gammaT(j,i)*sqrt(Tc[j]*Tc[i]);
                m.Yv(i,j) = 1.0/8.0*m.betaV(i,j)*m.gammaV(i,j)*POW3(cbrt(vc[i]) + cbrt(vc[j]));
                m.Yv(j,i) = 1.0/8.0*m.betaV(j,i)*m.gammaV(j,i)*POW3(cbrt(vc[j]) + cbrt(vc[i]));
            }
        }
        return m;
    }
    const TcVc m_Tcvc;
    const Matrices matrices;
public:
    GERG2004Reducing(const std::vector<std::string>& names): m_Tcvc(get_Tcvc(names)), matrices(get_matrices(names)) {}
    
    template<typename MoleFractions>
    auto Y(const MoleFractions& z, const std::vector<double>& Yc, const Eigen::ArrayXd& beta, const Eigen::ArrayXd& Yij){
        decltype(z[0]) sum1 = 0.0, sum2 = 0.0;
        std::size_t N = len(z);
        for (auto i = 0; i < N-1; ++i){
            sum1 += z[i]*z[i]*Yc[i];
            for (auto j = i+1; j < N; ++j){
                sum2 += 2.0*z[i]*z[j]*(z[i]+z[j])/(POW2(beta(i,j))*z[i]+z[j])*Yij(i,j);
            }
        }
        return forcceeval(sum1 + sum2);
    }
    
    template<typename MoleFractions>
    auto get_Tr(const MoleFractions& z){
        return Y(z, m_Tcvc.Tc_K, matrices.betaT, matrices.YT);
    }
        
    template<typename MoleFractions>
    auto get_rhor(const MoleFractions& z){
        return 1.0/Y(z, m_Tcvc.vc_m3mol, matrices.betaV, matrices.Yv);
    }
};

class GERG2004CorrespondingStatesTerm {

private:
    const std::vector<GERG2004PureFluidEOS> EOSs;
    auto get_EOS(const std::vector<std::string>& names){
        std::vector<GERG2004PureFluidEOS> theEOS;
        for (auto& name: names){
            theEOS.emplace_back(name);
        }
        return theEOS;
    }
public:
    GERG2004CorrespondingStatesTerm(const std::vector<std::string>& names) : EOSs(get_EOS(names)) {};
    
    auto size() const { return EOSs.size(); }

    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = std::common_type_t<decltype(tau), decltype(molefracs[0]), decltype(delta)>; // Type promotion, without the const-ness
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        if (N != size()){
            throw std::invalid_argument("wrong size");
        }
        for (auto i = 0; i < N; ++i) {
            alphar = alphar + molefracs[i] * EOSs[i].alphar(tau, delta);
        }
        return forceeval(alphar);
    }
};


// ***********************************************************
// ***********************************************************
//                          Departure
// ***********************************************************
// ***********************************************************


class GERG2004DepartureFunction {
private:
    const DepartureCoeffs dc;
public:
    GERG2004DepartureFunction() {};
    GERG2004DepartureFunction(const std::string& fluid1, const std::string& fluid2) : dc(get_departurecoeffs(fluid1, fluid2)){}

    template<typename TauType, typename DeltaType>
    auto alphar(const TauType& tau, const DeltaType& delta) const {
        using result = std::common_type_t<TauType, DeltaType>;
        result r = 0.0, lntau = log(tau);
        auto square = [](auto x) { return x * x; };
        if (getbaseval(delta) == 0) {
            for (auto i = 0; i < dc.n.size(); ++i) {
                r += dc.n[i] * exp(dc.t[i] * lntau - dc.eta[i] * square(delta - dc.epsilon[i]) - dc.beta[i] * (delta - dc.gamma[i]))*powi(delta, static_cast<int>(dc.d[i]));
            }
        }
        else {
            result lndelta = log(delta);
            for (auto i = 0; i < dc.n.size(); ++i) {
                r += dc.n[i] * exp(dc.t[i] * lntau + dc.d[i] * lndelta - dc.eta[i] * square(delta - dc.epsilon[i]) - dc.beta[i] * (delta - dc.gamma[i]));
            }
        }
        return forceeval(r);
    }
};

class GERG2004DepartureTerm {
private:
    const Eigen::ArrayXXd Fmat;
    const std::vector<std::vector<GERG2004DepartureFunction>> depmat;
    
    auto get_Fmat(const std::vector<std::string>& names){
        std::size_t N = names.size();
        Eigen::ArrayXXd mat(N, N); mat.setZero();
        for (auto i = 0; i < N; ++i){
            for (auto j = i; j < N; ++j){
                auto Fij = get_Fij(names[i], names[j]);
                if (Fij){
                    mat(i,j) = Fij.value();
                    mat(j,i) = mat(i,j); // Fij are symmetric
                }
            }
        }
        return mat;
    }
    auto get_depmat(const std::vector<std::string>& names){
        std::size_t N = names.size();
        std::vector<std::vector<GERG2004DepartureFunction>> mat(N);
        for (auto i = 0; i < N; ++i){
            std::vector<GERG2004DepartureFunction> row;
            for (auto j = 0; j < N; ++j){
                if (i != j && Fmat(i,j) != 0){
                    row.emplace_back(names[i], names[j]);
                }
                else{
                    row.emplace_back();
                }
            }
            mat.emplace_back(row);
        }
        return mat;
    }
public:
    GERG2004DepartureTerm(const std::vector<std::string>& names) : Fmat(get_Fmat(names)), depmat(get_depmat(names)) {};
    
    template<typename TauType, typename DeltaType, typename MoleFractions>
    auto alphar(const TauType& tau, const DeltaType& delta, const MoleFractions& molefracs) const {
        using resulttype = std::common_type_t<decltype(tau), decltype(molefracs[0]), decltype(delta)>; // Type promotion, without the const-ness
        resulttype alphar = 0.0;
        auto N = molefracs.size();
        if (N != Fmat.cols()){
            throw std::invalid_argument("wrong size");
        }
        
        for (auto i = 0; i < N; ++i){
            for (auto j = 0; j < N; ++j){
                auto Fij = Fmat(i,j);
                if (Fij != 0){
                    alphar += molefracs[i]*molefracs[j]*Fmat(i,j)*depmat[i][j].alphar(tau, delta);
                }
            }
        }
        return forceeval(alphar);
    }
};

class GERG2004ResidualModel{
public:
    GERG2004Reducing red;
            
    GERG2004CorrespondingStatesTerm corr;
    GERG2004DepartureTerm dep;
    GERG2004ResidualModel(const std::vector<std::string>& names) : red(names), corr(names), dep(names){}
    
    template<typename TType, typename RhoType, typename MoleFracType>
    auto alphar(const TType &T,
        const RhoType &rho,
        const MoleFracType& molefrac) const {
        auto Tred = forceeval(red.get_Tr(molefrac));
        auto rhored = forceeval(red.get_rhor(molefrac));
        auto delta = forceeval(rho / rhored);
        auto tau = forceeval(Tred / T);
        auto val = corr.alphar(tau, delta, molefrac) + dep.alphar(tau, delta, molefrac);
    }
};

} /* namespace GERG2004 */

namespace GERG2008{
//....
} /* namespace GERG2008 */

}
