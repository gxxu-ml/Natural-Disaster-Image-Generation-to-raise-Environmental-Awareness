#!/bin/bash
curl -L https://ucla.box.com/shared/static/szt6wcypjlqhj8d8885la5bd2jn50k8h --output aug_data.zip
ln -s . data40k
unzip aug_data.zip
rm aug_data.zip data40k
