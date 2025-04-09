SCENE_DIR="lego_data/lego/lego"
RESULT_DIR="daqian_test/lego"
SCENE_LIST="lego"
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do

        DATA_FACTOR=4

    echo "Running $SCENE"

    # Train without evaluation
    CUDA_VISIBLE_DEVICES=0 python train_lego.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR\
        --result_dir $RESULT_DIR/$SCENE/

    # Run evaluation and render for each checkpoint
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python train_lego.py default --disable_viewer --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir $SCENE_DIR \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done

# Print evaluation and training stats
for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="
    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS
        echo
    done

    echo "=== Train Stats ==="
    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS
        echo
    done
done
