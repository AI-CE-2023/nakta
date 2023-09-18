# 받기
# rsync -avz bhjpop:/runs/users/baehanjin/work/nakta ./ --exclude 'weights' --exclude '.git'

# 보내기
rsync -avz --delete ./ bhjpop:/runs/users/baehanjin/work/nakta --exclude 'weights' --exclude '.git'