/*
  Santa Clara University
  Paramdeep Singh
  last modified : 3/8/2025
 */

#include "mlp.h"


int main()
  {

    int i = 0;
    int j = 0;
    int arg_max = 0;
    int total_matches = 0;
    string filename;
    float sample;
    float label;

	//declare array to load test data
	act_t test_data_samples[NUM_TEST_SAMPLES][NUM_INPUTS];
	int test_labels[NUM_TEST_SAMPLES][NUM_OUTPUTS];

	int *output; //output buffer


	///////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////LOAD TEST DATA/////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != nullptr) {
                 std::cout << "Current directory: " << cwd << std::endl;
            } else {
                std::cerr << "Error getting current directory." << std::endl;
            }

			filename = TEST_DATA_FILE;
		    //cout << filename.c_str() << std::endl;
			ifstream test_data;
			test_data.open(filename.c_str());
			if (!test_data.is_open()){
				    cout << "Cannot open file " << TEST_DATA_FILE << "\n" ;
			}
            else{
                    cout << "Successfully Opened " << TEST_DATA_FILE << "\n";
            }      

			for (i= 0; i < NUM_TEST_SAMPLES ; i++){
				for(j=0 ; j < NUM_INPUTS; j++){
					test_data >> sample;
					test_data_samples[i][j] = sample ;
					//std::cout << sample << " ";
				}
				    //std::cout <<  std::endl;
			}

			cout << std::endl<<std::endl;
			cout << "Test Samples" << std::endl;
			cout << std::endl;

			//ensure the test samples array is populated correctly
			/*
			for (i= 0; i < NUM_TEST_SAMPLES ; i++){
				for(j=0 ; j < NUM_INPUTS; j++){
					  std::cout << test_data_samples[i][j] << " ";
					}
				    std::cout <<  std::endl;
				}
			*/
			test_data.close();
			///////////////////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////LOAD LABEL DATA/////////////////////////////////////////////////
			///////////////////////////////////////////////////////////////////////////////////////////////

						filename = LABEL_FILE;
					    //cout << filename.c_str() << std::endl;
						ifstream label_data;
						label_data.open(filename.c_str());
						if (!label_data.is_open()){
							    cout << "Cannot open Labels file " << LABEL_FILE << "\n" ;
							  }
                        else {
                             cout << "Successfully Opened " << LABEL_FILE << "\n";
                        }

						for (i= 0; i < NUM_TEST_SAMPLES ; i++){
							for(j=0 ; j < NUM_OUTPUTS; j++){
								label_data >> label;
								test_labels[i][j] = label;
							}

						}

						cout << std::endl<<std::endl;
						cout << "Test Labels" << std::endl;
						cout << std::endl;

						//ensure the test labels array is populated correctly

						/*
						for (i= 0; i < NUM_TEST_SAMPLES ; i++){
							for(j=0 ; j < NUM_OUTPUTS; j++){
								  std::cout  << test_labels[i][j] << " ";
								}
							    std::cout <<  std::endl;
							}
						*/




	///////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////EVALUATE MLP///////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////

			//initialize output buffer
			output = new(int);
			*output = 0;
			
            cout << "Outputs: " << endl;
        
			for (i= 0; i < NUM_TEST_SAMPLES ; i++){
				//Call MLP

				mlp_sparse(
					test_data_samples[i],
					output
					);

				arg_max = 0;
				for (j=0 ; j<NUM_OUTPUTS;j++){
                if(test_labels[i][j] == 1.0){
                	arg_max = j;
                	break;
                }
				}

				if(*output == arg_max) total_matches++;
				//cout << *output << endl;

			  }
			    cout << "Matches with ground truth labels = "  << total_matches << " out of " << NUM_TEST_SAMPLES << " Total samples" << std::endl;
			    cout << "Accuracy = " << ((total_matches*100)/NUM_TEST_SAMPLES)  << "% ";

	return 0;
  }

