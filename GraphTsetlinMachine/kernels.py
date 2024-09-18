# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

code_header = """
    #include <curand_kernel.h>
    
    #define INT_SIZE 32

    #define LA_CHUNKS (((LITERALS-1)/INT_SIZE + 1))
    #define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

    #if (LITERALS % 32 != 0)
    #define FILTER (~(0xffffffff << (LITERALS % INT_SIZE)))
    #else
    #define FILTER 0xffffffff
    #endif
"""

code_update = """
    extern "C"
    {
        // Counts number of include actions for a given clause
        __device__ inline int number_of_include_actions(unsigned int *ta_state)
        {
            int number_of_include_actions = 0;
            for (int k = 0; k < LA_CHUNKS-1; ++k) {
                unsigned int ta_pos = k*STATE_BITS + STATE_BITS-1;
                number_of_include_actions += __popc(ta_state[ta_pos]);
            }
            unsigned int ta_pos = (LA_CHUNKS-1)*STATE_BITS + STATE_BITS-1;
            number_of_include_actions += __popc(ta_state[ta_pos] & FILTER);

            return(number_of_include_actions);
        }

        // Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
        __device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active)
        {
            unsigned int carry, carry_next;
            int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
            carry = active;
            for (int b = 0; b < STATE_BITS; ++b) {
                if (carry == 0)
                    break;

                carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
                ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
                carry = carry_next;
            }

            if (carry > 0) {
                for (int b = 0; b < STATE_BITS; ++b) {
                    ta_state[id + b] |= carry;
                }
            }   
        }

        // Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
        __device__ inline void dec(unsigned int *ta_state, int clause, int chunk, unsigned int active)
        {
            unsigned int carry, carry_next;
            int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
            carry = active;
            for (int b = 0; b < STATE_BITS; ++b) {
                if (carry == 0)
                    break;
                carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
                ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
                carry = carry_next;
            }

            if (carry > 0) {
                for (int b = 0; b < STATE_BITS; ++b) {
                    ta_state[id + b] &= ~carry;
                }
            } 
        }

        __device__ inline void calculate_clause_output(curandState *localState, unsigned int *ta_state, int number_of_nodes, unsigned int *clause_output, int *clause_patch, int *X)
        {
            int output_one_patch_count = 0;
            *clause_patch = -1;
            *clause_output = 0;

            // Evaluate each patch (convolution)
            for (int patch = 0; patch < number_of_nodes; ++patch) {
                int patch_clause_output = 1;
                for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                    if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
                        patch_clause_output = 0;
                        break;
                    }
                }

                if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS - 1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
                    patch_clause_output = 0;
                }

                if (patch_clause_output) {
                    if (output_one_patch_count == 0) {
                        *clause_patch = patch;
                        *clause_output = 1;
                    } else if ((curand(localState) % (output_one_patch_count + 1)) == 0) {
                        *clause_patch = patch;
                    }
                    output_one_patch_count += 1;
                }
            }
        }

        __device__ inline void update_clause(curandState *localState, int *clause_weight, unsigned int *ta_state, int clause_output, int clause_patch, int *X, int y, int class_sum)
        {
            int target = 1 - 2*(class_sum > y);
            
            if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, CLASSES-1)) {
                return;
            }

            int sign = (*clause_weight >= 0) - (*clause_weight < 0);
        
            int absolute_prediction_error = abs(y - class_sum);
            if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
                if (target*sign > 0) {
                    int included_literals = number_of_include_actions(ta_state);

                    if (clause_output && abs(*clause_weight) < INT_MAX) {
                        (*clause_weight) += sign;
                    }

                    // Type I Feedback
                    for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                        // Generate random bit values
                        unsigned int la_feedback = 0;
                        for (int b = 0; b < INT_SIZE; ++b) {
                            if (curand_uniform(localState) <= 1.0/S) {
                                la_feedback |= (1 << b);
                            }
                        }

                        if (clause_output && included_literals <= MAX_INCLUDED_LITERALS) {
                            #if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                                inc(ta_state, 0, la_chunk, X[clause_patch*LA_CHUNKS + la_chunk]);
                            #else
                                inc(ta_state, 0, la_chunk, X[clause_patch*LA_CHUNKS + la_chunk] & (~la_feedback));
                            #endif

                            dec(ta_state, 0, la_chunk, (~X[clause_patch*LA_CHUNKS + la_chunk]) & la_feedback);
                        } else {
                            dec(ta_state, 0, la_chunk, la_feedback);
                        }
                    }
                } else if (target*sign < 0 && clause_output) {
                    // Type II Feedback

                    //if ((*clause_weight - sign) != 0) { 
                        (*clause_weight) -= sign;
                    //}

                    #if NEGATIVE_CLAUSES == 0
                        if (*clause_weight < 1) {
                            *clause_weight = 1;
                        }
                    #endif

                    for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                        inc(ta_state, 0, la_chunk, (~X[clause_patch*LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
                    }
                }
            }
        }

        // Evaluate example
        __global__ void evaluate(
            unsigned int *global_ta_state,
            int *clause_weights,
            int number_of_nodes,
            int graph_index,
            int *class_sum,
            int *X
        )
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            X = &X[graph_index * LA_CHUNKS];

            for (int clause = index; clause < CLAUSES; clause += stride) {
                unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

                int clause_output;
                for (int patch = 0; patch < number_of_nodes; ++patch) {
                    clause_output = 1;
                    for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                        if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
                            clause_output = 0;
                            break;
                        }
                    }

                    if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
                        clause_output = 0;
                    }

                    if (clause_output) {
                        break;
                    }
                }

                if (clause_output) {
                    for (int class_id = 0; class_id < CLASSES; ++class_id) {
                        int clause_weight = clause_weights[class_id*CLAUSES + clause];
                        atomicAdd(&class_sum[class_id], clause_weight);                 
                    }
                }
            }
        }

        // Update state of Tsetlin Automata team
        __global__ void update(
            curandState *state,
            unsigned int *global_ta_state,
            int *clause_weights,
            int number_of_nodes,
            int graph_index,
            int *class_sum,
            int *X,
            int *y,
            int example
        )
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            /* Copy state to local memory for efficiency */  
            curandState localState = state[index];

            X = &X[graph_index * LA_CHUNKS];

            // Calculate clause output first
            for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
                unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

                unsigned int clause_output;
                int clause_patch;
                calculate_clause_output(&localState, ta_state, number_of_nodes, &clause_output, &clause_patch, X);

                for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
                    int local_class_sum = class_sum[class_id];
                    if (local_class_sum > THRESHOLD) {
                        local_class_sum = THRESHOLD;
                    } else if (local_class_sum < -THRESHOLD) {
                        local_class_sum = -THRESHOLD;
                    }
                    update_clause(&localState, &clause_weights[class_id*CLAUSES + clause], ta_state, clause_output, clause_patch, X, y[example*CLASSES + class_id], local_class_sum);
                }
            }
        
            state[index] = localState;
        }
    }
"""

code_evaluate = """
    extern "C"
    {
        // Evaluate examples
        __global__ void evaluate(
            unsigned int *global_ta_state,
            int *clause_weights,
            int number_of_nodes,
            int graph_index,
            int *class_sum,
            int *X
        )
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            X = &X[graph_index * LA_CHUNKS];

            for (int clause = index; clause < CLAUSES; clause += stride) {
                unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

                int all_exclude = 1;
                for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                    if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
                        all_exclude = 0;
                        break;
                    }
                }

                if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
                    all_exclude = 0;
                }

                if (all_exclude) {
                    continue;
                }

                int clause_output;
                for (int patch = 0; patch < number_of_nodes; ++patch) {
                    clause_output = 1;
                    for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                        if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
                            clause_output = 0;
                            break;
                        }
                    }

                    if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
                        clause_output = 0;
                    }

                    if (clause_output) {
                        break;
                    }
                }

                if (clause_output) {
                    for (int class_id = 0; class_id < CLASSES; ++class_id) {
                        int clause_weight = clause_weights[class_id*CLAUSES + clause];
                        atomicAdd(&class_sum[class_id], clause_weight);                 
                    }
                }
            }
        }

  
    }
"""

code_prepare = """
    extern "C"
    {
        __global__ void prepare(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            curandState localState = state[index];

            for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
                for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
                    #if NEGATIVE_CLAUSES == 1
                        clause_weights[class_id*CLAUSES + clause] = 1 - 2 * (curand(&localState) % 2); // 1 - 2*(clause % CLASSES != class_id);
                    #else
                        clause_weights[class_id*CLAUSES + clause] = 1;
                    #endif
                }

                unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];
                for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                    for (int b = 0; b < STATE_BITS-1; ++b) {
                        ta_state[la_chunk*STATE_BITS + b] = ~0;
                    }
                    ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] = 0;
                }
            }

            state[index] = localState;
        }
    }
"""

code_transform = """
    extern "C"
    {
        // Transform examples

        __global__ void transform(
            unsigned int *included_literals,
            unsigned int *included_literals_length,
            int number_of_nodes,
            int *X,
            int *transformed_X
        )
        {   
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            int patch_chunks = (((number_of_nodes-1)/INT_SIZE + 1));
            unsigned int patch_filter;
            if (patch_chunks % 32 != 0) {
                patch_filter = (~(0xffffffff << (number_of_nodes % INT_SIZE)));
            } else {
                patch_filter = 0xffffffff;
            }

            for (int clause = index; clause < CLAUSES; clause += stride) {
                if (included_literals_length[clause] == 0) {
                    transformed_X[clause] = 0;
                    continue;
                }

                unsigned int clause_output = 0;
                for (int patch_chunk = 0; patch_chunk < patch_chunks-1; ++patch_chunk) {
                    clause_output = (~(0U));
                    for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
                        clause_output &= X[patch_chunk*LITERALS + included_literals[clause*LITERALS + literal]];
                    }

                    if (clause_output) {
                        break;
                    }
                }

                if (!clause_output) {
                    clause_output = patch_filter;
                    for (int literal = 0; literal < included_literals_length[clause]; ++literal) {
                        clause_output &= X[(PATCH_CHUNKS-1)*LITERALS + included_literals[clause*LITERALS + literal]];
                    }
                }

                transformed_X[clause] = clause_output;
            }
        }
    }
"""
