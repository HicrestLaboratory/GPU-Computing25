#!/bin/bash

binname="gemm"

nameO0="${binname}_O0"
if [[ ! -f "${nameO0}" ]]
then
    echo "Error: ${nameO0} does not exist"
    exit 1
fi

nameO1="${binname}_O1"
if [[ ! -f "${nameO1}" ]]
then
    echo "Error: ${nameO1} does not exist"
    exit 1
fi

nameO2="${binname}_O2"
if [[ ! -f "${nameO2}" ]]
then
    echo "Error: ${nameO2} does not exist"
    exit 1
fi

nameO3="${binname}_O3"
if [[ ! -f "${nameO3}" ]]
then
    echo "Error: ${nameO3} does not exist"
    exit 1
fi

blockbinname="block_gemm"

blockO0="${blockbinname}_O0"
if [[ ! -f "${blockO0}" ]]
then
    echo "Error: ${blockO0} does not exist"
    exit 1
fi

blockO1="${blockbinname}_O1"
if [[ ! -f "${blockO1}" ]]
then
    echo "Error: ${blockO1} does not exist"
    exit 1
fi

blockO2="${blockbinname}_O2"
if [[ ! -f "${blockO2}" ]]
then
    echo "Error: ${blockO2} does not exist"
    exit 1
fi

blockO3="${blockbinname}_O3"
if [[ ! -f "${blockO3}" ]]
then
    echo "Error: ${blockO3} does not exist"
    exit 1
fi


cmd="./${nameO0} $1 $1 $1 0"
echo -e "\n----------------------------------------------"
echo -e "\t${nameO0} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${nameO1} $1 $1 $1 0"
echo -e "\n----------------------------------------------"
echo -e "\t${nameO1} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${nameO2} $1 $1 $1 0"
echo -e "\n----------------------------------------------"
echo -e "\t${nameO2} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${nameO3} $1 $1 $1 0"
echo -e "\n----------------------------------------------"
echo -e "\t${nameO3} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

# --------------------------------------------------

cmd="./${blockO0} $1 $2"
echo -e "\n----------------------------------------------"
echo -e "\t${blockO0} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${blockO1} $1 $2"
echo -e "\n----------------------------------------------"
echo -e "\t${blockO1} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${blockO2} $1 $2"
echo -e "\n----------------------------------------------"
echo -e "\t${blockO2} execution"
echo -e "----------------------------------------------\n\n"
${cmd}

cmd="./${blockO3} $1 $2"
echo -e "\n----------------------------------------------"
echo -e "\t${blockO3} execution"
echo -e "----------------------------------------------\n\n"
${cmd}
