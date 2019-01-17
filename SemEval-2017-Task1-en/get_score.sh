#!/bin/bash

cd Output_Dev
echo VALIDATION SCORE
for f in ./STS.output*; do echo $f; perl correlation.pl STS.gs.dev.en-en.txt $f; done
cd ./../Output_Test
echo TESTING SCORE
for f in ./STS.output*; do echo $f; perl correlation.pl STS.gs.test.en-en.txt $f; done
