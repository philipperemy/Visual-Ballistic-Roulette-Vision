function [ ] = compute_gradient_func( full_video_path )
    [~, name, ~] = fileparts(full_video_path);
    fprintf('video_name = %s\n', name);
    % clear;clc;
    start_idx = 1;
    end_idx = 1e9;
    augmented_name = strcat('../output/', name, '/videos');
    prev_img = imread(sprintf(strcat(augmented_name, '/frames/output_%04d.png'), start_idx));
    for i = (start_idx+1):end_idx
        try
            img = imread(sprintf(strcat(augmented_name, '/frames/output_%04d.png'), i));
            gradient = img - prev_img;
            % imshow(gradient);
            output_filename = sprintf(strcat(augmented_name, '/gradients/output_%04d.png'), i);
            disp(output_filename);
            imwrite(gradient, output_filename);
            prev_img = img;
            fprintf('%d\n', i);
        catch e
            disp(e);
            exit();
        end
    end
    exit();

end

