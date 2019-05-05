#!/bin/bash
cat $1 | grep -v '^c' | sed 's/p [^0-9]*//' | sed 's/e //' > $2