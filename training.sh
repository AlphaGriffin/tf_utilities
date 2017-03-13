
DELAY=5

# Logging these events
LOG=./update.log
exec &> >(tee -ia update.log)

TODAY=$(date)
HOST=$(hostname)
clear
echo "-----------------------------------------------------"
echo "Attempting to update this set of Github Libs"
echo "Date: $TODAY"
echo "Host: $USER@$HOST"
echo "-----------------------------------------------------"


OPTIONS="Check_Folders About Quit"
#TEST="'ls -d */' Hello Quit"
select opt in $OPTIONS; do
   if [ "$opt" = "Check_Folders" ]; then
    grep_status
   elif [ "$opt" = "About" ]; then
	echo """Alphagriffin.com"""
   elif [ "$opt" = "Quit" ]; then
	echo done
	exit
   else
	echo bad option
   fi
done
