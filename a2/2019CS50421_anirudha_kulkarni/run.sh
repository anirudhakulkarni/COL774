# first argument is question number
# second argument is relative path of train file
# third argument is relative path of test file
# bash run.sh 2 "./dataset/train.csv" "./dataset/test.csv" d
# bash run.sh 1 "./dataset/Music_Review_train.json" "./dataset/Music_Review_test.json" g 
# extract arguments
# python svm-final.py "./dataset/train_.csv" "./dataset/test_.csv" 1 a 
question=$1
train_path=$2
test_path=$3
if [ "$question" == "1" ];then
    part_num=$4
    python nb.py $train_path $test_path $part_num
elif [ "$question" == "2" ];then
    class=$4
    part_num=$5
    python3 svm-final.py $train_path $test_path $class $part_num
fi