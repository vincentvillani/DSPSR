#!/bin/bash
./dspsr -F 32:D -E 1644-4559.eph -P 1644-4559.polyco 1644-4559.cpsr2 -V -L0.5 -cuda 0 -covar &>dspsr.log
#./dspsr -U 1024 -F 32:D -E 1644-4559.eph -P 1644-4559.polyco 1644-4559.cpsr2 -v -L0.5 -cuda 0 -covar #&>dspsr.log
