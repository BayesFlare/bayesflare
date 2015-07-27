#!/bin/bash
echo "runningmedian"
time python filter_tester.py runningmedian
echo "highpassfilter"
time python filter_tester.py highpassfilter
echo "savitzkygolay"
time python filter_tester.py savitzkygolay
echo "supersmoother"
time python filter_tester.py supersmoother
