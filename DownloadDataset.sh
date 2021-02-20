wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B8-rUzbwVRk0c054eEozWG9COHM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B8-rUzbwVRk0c054eEozWG9COHM" -O market1501.zip && rm -rf /tmp/cookies.txt

unzip market1501.zip

mv Market-1501-v15.09.15 market1501

rm market1501.zip