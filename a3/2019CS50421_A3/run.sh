# first argument: question number
# second argument: train path
# third argument: test path
# fourth argument: validation path
# fifth argument: part number

question=$1
train=$2
test=$3
# ./run.sh 1 "./bank_dataset/bank_train.csv" "./bank_dataset/bank_test.csv" "./bank_dataset/bank_val.csv" a
# ./run.sh 2 "./poker_dataset/poker-hand-training-true.data" "./poker_dataset/poker-hand-testing.data" e

if [ $question == '1' ];
then
    validation=$4
    part=$5
    python3 dt-main.py $train $test $validation $part "a"
    python3 dt-main.py $train $test $validation $part "b"
    
elif [ $question == '2' ];
then
    part=$4
    python nn-main.py $train $test $part
fi