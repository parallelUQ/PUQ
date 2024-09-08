
# PUQ Test Runner

# Test options
export TEST_LIST=test_*.py    # Provide a list of tests #unordered # override with -y

export RUN_COV_TESTS=true     # Provide a coverage report

export RUN_UQ_TESTS=true


# Try using git root dir
root_found=false
ROOT_DIR=$(git rev-parse --show-toplevel) && root_found=true

# If not found using git - try search up tree for setup.py
if [[ $root_found == "false" ]]; then
    search_dir=`pwd`
    search_for="setup.py"
    while [ "$search_dir" != "/" ]
    do
        file_found=$(find_file_in_dir $search_dir $search_for)
        if [[ $file_found = "true" ]]; then
            ROOT_DIR=$search_dir
            root_found=true
            break;
        fi;
        search_dir=`dirname "$search_dir"`
    done
fi;

# Test Directories - all relative to project root dir
export TEST_SUBDIR_DES=$ROOT_DIR/tests/unit_tests


COV_LINE_SERIAL=''
if [ $RUN_COV_TESTS = "true" ]; then
   COV_LINE_SERIAL='--cov --cov-report html:cov_html'
fi;

# Run Tests -----------------------------------------------------------------------

echo -e "\n************** Running: PUQ Test-Suite **************\n"
for DIR in $TEST_SUBDIR_DES
do
  cd $DIR
  for TEST_SCRIPT in $TEST_LIST
  do
    pytest $TEST_SCRIPT $COV_LINE_SERIAL
  done
done