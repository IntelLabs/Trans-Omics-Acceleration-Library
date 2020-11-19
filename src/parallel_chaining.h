#include<iostream>
#include <stdio.h>
#include<string.h>
using namespace std;


static const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	-1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

static inline int ilog2_32(uint32_t v)
{
	uint32_t t, tt;
	if ((tt = v>>16)) return (t = tt>>8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
	return (t = v>>8) ? 8 + LogTable256[t] : LogTable256[v];
}

class anchor_t {
	public:
		uint64_t r;
		int32_t q;
		int32_t l;
		anchor_t(){}

		anchor_t(uint64_t x, int32_t y, int32_t length){
			r = x; q = q; l = length;
		}	
};


class dp_chain {
	public: 
	int max_dist_x, max_dist_y, bw, max_skip, max_iter, is_cdna, n_segs;
	float gap_scale;

	dp_chain(){}

	void test(){
		cout<<"function called\n";
		printf("hyper-parameters: %d %d %d %d %d %f %d %d\n",max_dist_x, max_dist_y, bw, max_skip, max_iter, gap_scale, is_cdna, n_segs);
	}

	dp_chain(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, float gap_scale, int is_cdna, int n_segs){
 		this->max_dist_x = max_dist_x;
		this->max_dist_y = max_dist_y;
		this->bw = bw;
		this->max_skip = max_skip;
		this->max_iter = max_iter;
		this->gap_scale = gap_scale;
		this->is_cdna = is_cdna;
		this->n_segs = n_segs;

	}

	int32_t get_gap_cost(anchor_t a, anchor_t b, float avg_qspan, float gap_scale){
		int64_t dr = a.r - b.r;
		int32_t dq = a.q - b.q;

		bool flag = true;
		bool is_cdna = false;	
		int32_t dd = dr > dq? dr - dq : dq - dr; //smk: dd = |dr-dq|;
		

		int32_t log_dd = dd? ilog2_32(dd) : 0;
		int32_t gap_cost = 0, sc = 0;


		if (is_cdna || !flag) {
			int c_log, c_lin;
			c_lin = (int)(dd * .01 * avg_qspan);
			c_log = log_dd;
			if (!flag && dr == 0) 
				++sc; // possibly due to overlapping paired ends; give a minor bonus
			else if (dr > dq || !flag) 
				gap_cost = c_lin < c_log? c_lin : c_log;
			else 
				gap_cost = c_lin + (c_log>>1);
		} 
		else 
			gap_cost = (int)(dd * .01 * avg_qspan) + (log_dd>>1);

		
		gap_cost = ((double)gap_cost * gap_scale + .499); //smk: does not match with the paper

		gap_cost = gap_cost - sc;			

		return gap_cost;
	}
	void mm_dp_lib(int64_t n, anchor_t *anchors, int32_t* &f, int32_t* &p, int32_t* &v)
	{ 

		int32_t *t;
		int64_t i, j, st = 0;
		uint64_t sum_qspan = 0;
		float avg_qspan;

		f = (int32_t*)malloc(n * 4);
		p = (int32_t*)malloc(n * 4);
		t = (int32_t*)malloc(n * 4);
		v = (int32_t*)malloc(n * 4);
		memset(t, 0, n * 4);

		for (i = 0; i < n; ++i) sum_qspan += anchors[i].l;
		avg_qspan = (float)sum_qspan / n;
		
		// fill the score and backtrack arrays
		for (i = 0; i < n; ++i) {
			uint64_t ri = anchors[i].r;
			int32_t qi = anchors[i].q;
			int32_t q_span = anchors[i].l;
		
			//Anchor <ri, qi, q_span>

			int64_t max_j = -1;
			int32_t max_f = q_span, n_skip = 0;
			

			while (st < i && ri > anchors[st].r + max_dist_x) ++st; //smk: predecessor's position is too far
			if (i - st > max_iter) st = i - max_iter; //smk: predecessor's index is too far
			
			int32_t sidi = 0;//(a[i].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;

			for (j = i - 1; j >= st; --j) {

				int64_t dr = ri - anchors[j].r;
				int32_t dq = qi - anchors[j].q;

				int32_t dd, sc = 0;

				int32_t sidj = 0;//(a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;

				if ((sidi == sidj && dr == 0) || dq <= 0) continue; // don't skip if an anchor is used by multiple segments; see below
				
				if ((sidi == sidj && dq > max_dist_y) || dq > max_dist_x) continue;

				dd = dr > dq? dr - dq : dq - dr; //smk: dd = |dr-dq|;
				if (sidi == sidj && dd > bw) continue; 

				if (n_segs > 1 && !is_cdna && sidi == sidj && dr > max_dist_y) continue;



				int32_t gap_cost = get_gap_cost(anchors[i], anchors[j], avg_qspan, gap_scale);
		

				int32_t min_d = dq < dr? dq : dr;
				int32_t alpha = min_d > q_span? q_span : min_d; //alpha = min(dq, dr, q_span);
				
				sc = f[j] + alpha - (int) gap_cost;

				if (sc > max_f) {
					max_f = sc, max_j = j;
					if (n_skip > 0) --n_skip;
				} else if (t[j] == i) {
					if (++n_skip > max_skip)
						break;
				}
				if (p[j] >= 0) t[p[j]] = i;
			}
			f[i] = max_f, p[i] = max_j;
			v[i] = max_j >= 0 && v[max_j] > max_f? v[max_j] : max_f; // v[] keeps the peak score up to i; f[] is the score ending at i, not always the peak
		}
		free(t);
	}

};


/*

void mm_dp_lib(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, float gap_scale, int is_cdna, int n_segs, int64_t n, mm128_t *a, anchor_t *anchors, int32_t* &f, int32_t* &p, int32_t* &v)
{ // TODO: make sure this works when n has more than 32 bits

	int32_t *t;
	int64_t i, j, st = 0;
	uint64_t sum_qspan = 0;
	float avg_qspan;

	f = (int32_t*)malloc(n * 4);
	p = (int32_t*)malloc(n * 4);
	t = (int32_t*)malloc(n * 4);
	v = (int32_t*)malloc(n * 4);
	memset(t, 0, n * 4);

	for (i = 0; i < n; ++i) sum_qspan += a[i].y>>32&0xff;
	avg_qspan = (float)sum_qspan / n;
	
	// fill the score and backtrack arrays
	for (i = 0; i < n; ++i) {
		uint64_t ri = anchors[i].r;
		int32_t qi = anchors[i].q;
		int32_t q_span = anchors[i].l;
	
		//Anchor <ri, qi, q_span>

		int64_t max_j = -1;
		int32_t max_f = q_span, n_skip = 0;
		

		while (st < i && ri > anchors[st].r + max_dist_x) ++st; //smk: predecessor's position is too far
		if (i - st > max_iter) st = i - max_iter; //smk: predecessor's index is too far
		
		int32_t sidi = 0;//(a[i].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;

		for (j = i - 1; j >= st; --j) {

			int64_t dr = ri - anchors[j].r;
			int32_t dq = qi - anchors[j].q;

			int32_t dd, sc = 0;

			int32_t sidj = 0;//(a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;

			if ((sidi == sidj && dr == 0) || dq <= 0) continue; // don't skip if an anchor is used by multiple segments; see below
			
			if ((sidi == sidj && dq > max_dist_y) || dq > max_dist_x) continue;

			dd = dr > dq? dr - dq : dq - dr; //smk: dd = |dr-dq|;
			if (sidi == sidj && dd > bw) continue; 

			if (n_segs > 1 && !is_cdna && sidi == sidj && dr > max_dist_y) continue;



			int32_t gap_cost = get_gap_cost(anchors[i], anchors[j], avg_qspan, gap_scale);
	

			int32_t min_d = dq < dr? dq : dr;
			int32_t alpha = min_d > q_span? q_span : min_d; //alpha = min(dq, dr, q_span);
			
			sc = f[j] + alpha - (int) gap_cost;

			if (sc > max_f) {
				max_f = sc, max_j = j;
				if (n_skip > 0) --n_skip;
			} else if (t[j] == i) {
				if (++n_skip > max_skip)
					break;
			}
			if (p[j] >= 0) t[p[j]] = i;
		}
		f[i] = max_f, p[i] = max_j;
		v[i] = max_j >= 0 && v[max_j] > max_f? v[max_j] : max_f; // v[] keeps the peak score up to i; f[] is the score ending at i, not always the peak
	}
	free(t);
}

*/
