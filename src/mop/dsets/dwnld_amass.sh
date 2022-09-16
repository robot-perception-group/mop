#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
urld () { [[ "${1}" ]] || return 1; : "${1//+/ }"; echo -e "${//%/\\x}"; }

read -p "Username:" username
read -p "Password:" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&sfile=DanceDB.tar.bz2' -O 'DanceDB.tar.bz2'