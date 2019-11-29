sed -i "s/^.*:/.\/inspect_autoenc.py -title/" $1
sed -i "s/(/-D /" $1
sed -i "s/,/ -F/" $1
sed -i "s/)/ -article $2/" $1

