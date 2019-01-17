#!/bin/bash
echo START RUN PYTHON SCRIPTS
python sem1-main.py -embed glove -d 50 -p ./../../Perso/Pretrained-Embedding/GloVe/
python sem1-main.py -embed glove -d 100 -p ./../../Perso/Pretrained-Embedding/GloVe/
python sem1-main.py -embed glove -d 200 -p ./../../Perso/Pretrained-Embedding/GloVe/
python sem1-main.py -embed miniNumberbatch -p ./../../Semantic_Network/ConceptNet/17.06/mini.h5
python sem1-main.py -embed elmo
python sem1-main.py -embed infersent -p ./../../Perso/InferSent/ -v 1 -ep ./../../Perso/Pretrained-Embedding/GloVe/glove.840B.300d.txt
