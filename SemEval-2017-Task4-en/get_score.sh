#!/bin/bash
cd SemEval2016_task4_submissions_and_scores/_scripts/
for f in ./../../Output/*.txt; do perl SemEval2017_task4_test_scorer_subtaskA.pl ./../../2017_English_final/DOWNLOAD/Subtask_A/twitter-2016test-A.txt $f;
done
