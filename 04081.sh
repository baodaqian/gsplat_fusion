SCENE_DIR="radar_data"
RESULT_DIR="daqian_test/radar_cube"
SCENE_LIST="cube_nerf"
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do

        DATA_FACTOR=4

    echo "Running $SCENE"

    # Train without evaluation
    CUDA_VISIBLE_DEVICES=0 python train_cube.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir radar_data/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # Run evaluation and render for each checkpoint
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python train_cube.py default --disable_viewer --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir radar_data/$SCENE/ \
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
