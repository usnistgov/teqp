// Prototypes of the functions exposed by the shared library
extern "C" int build_model(const char* j, char* uuid, char* errmsg, int errmsg_length);
extern "C" int free_model(const char* uid, char* errmsg, int errmsg_length);
extern "C" int get_Arxy(const char* uid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length);

#include <valarray>;

int main(){
	const char* model = R"(  
        {
            "kind": "SRK", 
            "model": {
                "Tcrit / K": [190], 
                "pcrit / Pa": [3.5e6], 
                "acentric": [0.11]
            }
        }
    )";
	// Build the model
    char uid[40], errstr[200];
	int errcode = build_model(model, uid, errstr, 200);

    // Call the model
    int NT = 0, ND = 1;
    double T = 300, rho = 0.5, out = -1;
    std::valarray<double> z(1.0, 1);
    int errcode2 = get_Arxy(uid, NT, ND, T, rho, &(z[0]), z.size(), &out, errstr, 200);

    return EXIT_SUCCESS;
}