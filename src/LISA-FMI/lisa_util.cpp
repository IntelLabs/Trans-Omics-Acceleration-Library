#include "lisa_util.h"

SMEM_out::SMEM_out(int _id, int _q_l, int _q_r, index_t _ref_l, index_t _ref_r) {
	id = _id;
	q_l = _q_l;
	q_r = _q_r;
	ref_l = _ref_l;
	ref_r = _ref_r;
}

Output::Output(int a) { 
	id  = a;
}

void Info::set(int a, int b, index_t c, index_t d) {
	l = a; r = b; intv = make_pair(c,d);
}


uint64_t Info::get_enc_str(){

	uint64_t nxt_ext = 0;
	int i = l - 21; //K;
	//TODO: hard coded K
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);

	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);

	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);

	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);
	nxt_ext = (nxt_ext<<2) | dna_ord(p[i++]);

	return nxt_ext;
}


void load(string filename, vector<char*> ptrs, vector<size_t> sizes) {
	ifstream instream(filename.c_str(), ifstream::binary);
	instream.seekg(0);
	assert(ptrs.size() == sizes.size());
	for(size_t i=0; i<ptrs.size(); i++) {
		instream.read(ptrs[i], sizes[i]);
	}
	instream.close();
}

void save(string filename, vector<char*> ptrs, vector<size_t> sizes) {
	ofstream outstream(filename.c_str(), ofstream::binary);
	outstream.seekp(0);
	assert(ptrs.size() == sizes.size());
	for(size_t i=0; i<ptrs.size(); i++) {
		outstream.write(ptrs[i], sizes[i]);
	}
	outstream.close();
}

int64_t FCLAMP(double inp, double bound) {
	if (inp < 0.0) return 0;
	return (inp > bound ? bound : (int64_t)inp);
}

string get_abs_path(string path){
	/* Get absolute path to the reference sequence file */
	/*   
		 srand(time(0));
		 int random_num = rand() % 10000;    
		 string temp_filename = path + "_ref_path" + to_string(random_num);

		 string cmd = "readlink -f "+ path + " > " + temp_filename;
		 system(cmd.c_str());
		 ifstream f_abs_path(temp_filename);
		 fprintf(stderr, "Relative ref path: %s\n", path.c_str());
		 string abs_path;
		 f_abs_path>>abs_path;
		 fprintf(stderr, "Absolute ref path: %s\n", abs_path.c_str());

		 f_abs_path.close();

		 cmd = "rm " + temp_filename;
		 system(cmd.c_str());
		 */

	string abs_path;

	if (path[0] == '/') return path;

	abs_path = (string) get_current_dir_name();
	abs_path = abs_path + "/" + path;
	fprintf(stderr, "Absolute ref path with get_current_dir: %s\n", abs_path.c_str());

	return abs_path;



}
string get_abs_location(string path) {

	string abs_path = get_abs_path(path);	
	string directory;
	const size_t last_bslash_idx = abs_path.rfind('/');
	if (std::string::npos != last_bslash_idx)
	{
		directory = abs_path.substr(0, last_bslash_idx);
	}
	fprintf(stderr, "Absolute directory path: %s\n", directory.c_str());

	return directory;

}


