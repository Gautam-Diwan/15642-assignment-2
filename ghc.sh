#!/usr/bin/env bash

for i in $(seq 1 86); do

  (

    host="ghc${i}.ghc.andrew.cmu.edu"

    if ping -c 1 -W 1 "$host" &> /dev/null; then

      echo "UP   - $host"

    fi

  ) &

done

wait