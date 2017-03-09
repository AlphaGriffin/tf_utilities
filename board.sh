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
	figlet "Tensorflow" "    Utilities"
	echo "-----------------------------------------------------"
	echo "Date: $TODAY"
	echo "Host: $USER@$HOST"
	echo "-----------------------------------------------------"
}

function different_options() {
	all_done=0
	while (( !all_done )); do
			clear;
			show_header;
			options=("Check Connectivity" "Show Header")

			echo "Choose an option:"
			select opt in "${options[@]}"; do
					case $REPLY in
							1) checkConnectivity; break ;;
							2) show_header; break ;;
							*) echo "What's that?" ;;
					esac
			done

			echo "Doing other things...";

			echo "Are we done?"
			select opt in "Yes" "No"; do
					case $REPLY in
							1) all_done=1; break ;;
							2) break ;;
							*) echo "Look, it's a simple question..." ;;
					esac
			done
	done
	
}


function options_screen() {
	# a Space Sperated List of Options that will be given
	OPTIONS="Check_Folders checkConnectivity run_MupenTest About Quit";
	select opt in $OPTIONS; do
		if [ "$opt" = "Check_Folders" ]; then
			# grep_status
			clear;
			figlet "Check Git status";
		
		elif [ "$opt" = "checkConnectivity" ]; then
			clear;
			figlet "Check Connectivity";
			echo "AlphaGriffin.com online: $(checkConnectivity;)"
		
		elif [ "$opt" = "About" ]; then
			sl -Fec;
			clear;
			figlet """Alphagriffin.com""";
		
		elif [ "$opt" = "run_MupenTest" ]; then
			sl -Fec;
			clear;
			figlet "Mupen Test";
			python mupentest.py
		
		elif [ "$opt" = "Quit" ]; then
			figlet "done";
			exit;
		
		else
			clear;
			show_header;
			figlet "bad option";
		fi
	done;
}

# This is the program Launch Begining
show_header
#sl
#options_screen
options_screen
