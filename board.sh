#!/bin/bash
#"""
#DummyScript.com 2017
#Created on Tue Feb 28 00:01:38 2017
#@author: eric
#"""
#"""
#usage: tensorboard [-h] [--logdir LOGDIR] [--debug [DEBUG]] [--nodebug]
#                   [--host HOST] [--inspect [INSPECT]] [--noinspect]
#                   [--tag TAG] [--event_file EVENT_FILE] [--port PORT]
#                   [--purge_orphaned_data [PURGE_ORPHANED_DATA]]
#                   [--nopurge_orphaned_data]
#                   [--reload_interval RELOAD_INTERVAL]
#"""

# tensorboard --logdir ./train_logs --reload_interval 25


DELAY=5

# Logging these events
LOG=./tensorflow.log
exec &> >(tee -ia tensorflow.log)

TODAY=$(date)
HOST=$(hostname)

function printResult() {
	[[ $1 -ne 0 ]] && echo -e "\e[1;31m[FAIL]\e[0m" || echo -e "\e[1;32m[OK]\e[0m\e[37m"
	return 0
}

function checkConnectivity() {
	ping -c 1 -w $DELAY alphagriffin.com >> $LOG 2>&1
	local BUFFER=$?
	printResult $BUFFER
	# [[ $BUFFER -ne 0 ]] && exit 1
	return 0
}

# Takes 1 arguement as DIR for this git_update
function git_update() {
    if git status | grep -q up-to-date; then
			echo "Repo: $1 is up to date"
    else
        # if pulled latest
        if git pull | grep -q Already; then
            echo "Repo: $1 has been updated"
        else
            # if we are failing at this...
            echo "failed to pull new Repo for $1"
        fi
	fi

}
function grep_status() {
	for i in $( ls -d */); do
		# go into dir
		cd $i 
		# if git is up to date
		if git status | grep -q up-to-date; then
			echo "Repo: $i is up to date"
		else
			# if pulled latest
			if git pull | grep -q Already; then
				echo "Repo: $i has been updated"
			else
				echo "failed to pull new Repo for $i"
			fi
		fi
		# back out of each dir as you work through
		cd ..
	done 
}

function show_header() {
	clear
	figlet "tf_utilities"
	echo "-----------------------------------------------------"
	echo "Date: $TODAY"
	echo "Host: $USER@$HOST"
	echo "-----------------------------------------------------"
}

function start_tensorboard() {
    tensorboard --logdir $PWD/train_logs --reload_interval 25 &
    #local BUFFER=$?
	#printResult $BUFFER
    #return X
}

# takes one arguement to kill the process
function stop_tensorboard() {
    kill $1
    local BUFFER=$?
	printResult $BUFFER
}

function options_screen() {
	# a Space Sperated List of Options that will be given
	options=("Git Update" "Check Connectivity" "Run Mupen Test" "Start Tensorbaord" "Stop Tensorbaord" "Show Log" "About" "Quit")

	# global variables
	all_done=0
	on_exit=0
	tensorboard_pid=0

	while (( !all_done )); do
	    while (( !on_exit )); do
            clear;
            show_header;
            # select opt in $OPTIONS; do
            select opt in "${options[@]}"; do
                if [ "$opt" = "Git Update" ]; then
                    # grep_status
                    clear;
                    figlet "Check Git status";
                    git_update $PWD;
                    break;

                elif [ "$opt" = "Check Connectivity" ]; then
                    # use this to check if a webpage is up
                    clear;
                    figlet "Check Connectivity";
                    echo "AlphaGriffin.com online: $(checkConnectivity;)"
                    break;

                elif [ "$opt" = "Show Log" ]; then
                    # tail the log
                    clear;
                    figlet "Show Log";
                    tail -n 3 $PWD/tensorflow.log;
                    break;

                elif [ "$opt" = "Start Tensorbaord" ]; then
                    # start the tensorboard program
                    clear;
                    figlet "Tensorboard";
                    stop_tensorboard $tensorboard_pid
                    echo "Tensorboard started properly"
                    break;

                elif [ "$opt" = "Stop Tensorbaord" ]; then
                    clear;
                    figlet "Tensorboard";
                    tensorboard_pid= start_tensorboard &
                    echo "Tensorboard started properly"
                    break;

                elif [ "$opt" = "Run Mupen Test" ]; then
                    clear;
                    figlet "Mupen Test";
                    python mupentest.py;
                    break;

                elif [ "$opt" = "About" ]; then
                    clear;
                    figlet """Alphagriffin.com""";
                    break;

                elif [ "$opt" = "Quit" ]; then
                    figlet "done?";
                    on_exit=1
                    break;

                else
                    figlet "bad option";
                fi
            done
        done

        show_header;
        echo "Are you really done?"
        select opt in "Yes" "No"; do
                case $REPLY in
                        1) all_done=1; break ;;
                        2) break ;;
                        *) echo "Look, it's a simple question..." ;;
                esac
        done
    done
}

# This is the program Launch Begining
show_header
options_screen