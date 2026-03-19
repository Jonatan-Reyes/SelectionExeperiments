#!/bin/bash
key=~/.ssh/id_rsa
user=jorm@di.ku.dk
erdadir=./GAIN/AnnonomizedPresagereEndoscopies/
mnt=./Data_Erda
if [ -f "$key" ]
then
    mkdir -p ${mnt}
    echo "test"
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval\
=15,ServerAliveCountMax=3 -o IdentityFile=${key}
else
    echo "'${key}' is not an ssh key"
fi
