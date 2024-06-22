
EXPORT_CODE int CONVENTION build_model(const char* j, long long int* uuid, char* errmsg, int errmsg_length);

EXPORT_CODE int CONVENTION free_model(const long long int uuid, char* errmsg, int errmsg_length);

EXPORT_CODE int CONVENTION get_Arxy(const long long int uuid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length);

EXPORT_CODE int CONVENTION get_ATrhoXi(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_ATrhoXiXj(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_ATrhoXiXjXk(const long long int uuid, const double T, const int NT, const double rhomolar, const int ND, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_AtaudeltaXi(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_AtaudeltaXiXj(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_AtaudeltaXiXjXk(const long long int uuid, const double tau, const int Ntau, const double delta, const int Ndelta, const double* molefrac, const int Ncomp, const int i, const int NXi, const int j, const int NXj, const int k, const int NXk, double *val, char* errmsg, int errmsg_length) ;

EXPORT_CODE int CONVENTION get_dmBnvirdTm(const long long int uuid, const int Nvir, const int NT, const double T, const double* molefrac, const int Ncomp, double* val, char* errmsg, int errmsg_length) ;
