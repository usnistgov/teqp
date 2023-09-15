// Prototypes of the functions exposed by the shared library
extern "C" int build_model(const char* j, long long int* uuid, char* errmsg, int errmsg_length);
extern "C" int free_model(const long long int uid, char* errmsg, int errmsg_length);
extern "C" int get_Arxy(const long long int uid, const int NT, const int ND, const double T, const double rho, const double* molefrac, const int Ncomp, double *val, char* errmsg, int errmsg_length);

#include <valarray>

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
    char errstr[200];
    long long int uid = -1;
	int errcode = build_model(model, &uid, errstr, 200);
    if (errcode != 0){
        return EXIT_FAILURE;
    }

    // Call the model
    int NT = 0, ND = 1;
    double T = 300, rho = 0.5, out = -1;
    std::valarray<double> z(1.0, 1);
    int errcode2 = get_Arxy(uid, NT, ND, T, rho, &(z[0]), static_cast<int>(z.size()), &out, errstr, 200);
    if (errcode2 != 0){
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
