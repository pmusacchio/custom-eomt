#!/bin/bash

srun --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G --gres=gpu:a100:$1 --partition=oermannlab --exclude=a100-8003 --time 24:00:00 --job-name dev --pty bash
