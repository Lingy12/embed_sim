EVAL_SET='test,short,color'


python multilang_eval_batched.py data/sample_test.json /home/shared_LLMs/SEA_LLMs/Sailor-4B-Chat $EVAL_SET /home/shared_LLMs/SEA_LLMs/Tamil_Pretraining_V0.1/checkpoint-7250
python multilang_eval_batched.py data/sample_test.json /home/shared_LLMs/SEA_LLMs/Sailor-4B-Chat $EVAL_SET /home/shared_LLMs/SEA_LLMs/Tamil_Pretraining_V0.1/checkpoint-3750
python multilang_eval_batched.py data/sample_test.json /home/shared_LLMs/SEA_LLMs/Sailor-4B-Chat $EVAL_SET /home/shared_LLMs/SEA_LLMs/Tamil_Pretraining_V0.1/checkpoint-5750
python multilang_eval_batched.py data/sample_test.json /home/shared_LLMs/SEA_LLMs/Sailor-4B-Chat $EVAL_SET