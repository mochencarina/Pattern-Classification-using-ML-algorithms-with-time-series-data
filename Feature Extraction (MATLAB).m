clear all
clc

Dir = 'E:\Fall 2018\SYSC 5405\Project\train';

T = ["control", "als", "hunt", "park"];

VEC = [];
%Read additional train information
clc
filename = 'E:\Fall 2018\SYSC 5405\Project\train\train.txt';
delimiter = '\t';
startRow = 2;
formatSpec = '%s%s%s%s%s%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
train = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

for i = 1:20
    for j = 1:4
        Type = T(j);
        
        Filename = sprintf('%s%d%s',Type, i, '.tsv');
        A = exist(Filename, 'file');
        if A == 2
            B = dlmread(Filename,'\t');
            
            % Preprocessomg data
            % 1st column for left, 2nd column for right foot force
            % Smoothing nosisy data with moving average
            k = 101;
            B(1:5999, :) = [];
            B(:, 1) = smoothdata(B(:, 1), 'sgolay', k);
            B(:, 2) = smoothdata(B(:, 2), 'sgolay', k);
            figure(1)
            
            %--------------- Time domain features ---------------%
            % Normal attributes: rms, std dev, correlation 
            Fs = 300;
            
            % RMS of the whole training data in time domain             
            RMS_l = rms(B(:, 1));
            RMS_r = rms(B(:, 2));
            
            % Standard deviation of time series data
            Std_dev_l = std(B(:, 1));
            Std_dev_r = std(B(:, 2));
            
            % Correlation between left and right samples
            correlation = corrcoef(B(:, 1), B(:, 2));
            COR = correlation(1, 2);

            %----------------------- Peak detection --------------------%
            % For left foot force
            [pks_l, pks_locs_l, pks_w_l, pks_p_l] = findpeaks(B(:, 1),Fs,'MinPeakDistance',1.16);                     
            mean_dist_pks_l = mean(diff(pks_locs_l)); % Mean distance among peaks                    
            mean_val_pks_l = mean(pks_l);  % Mean value among peaks  
            no_of_pks_l = length(pks_l); % Number of peaks
            
            % For right foot force
            [pks_r, pks_locs_r, pks_w_r, pks_p_r] = findpeaks(B(:, 2),Fs,'MinPeakDistance',1.16);
            mean_dist_pks_r = mean(diff(pks_locs_r));
            mean_val_pks_r = mean(pks_r);
            no_of_pks_r = length(pks_r);
            
            % RMS of peaks
            RMS_pks_l = rms(pks_l);
            RMS_pks_r = rms(pks_r);

            %---------------------- Bottom detection -------------------%
            % For left foot force
            [bot_l, bot_locs_l, bot_w_l, bot_p_l] = findpeaks(-B(:, 1),Fs,'MinPeakDistance',1.16);                     
            mean_dist_bot_l = mean(diff(bot_locs_l));% Mean distance among bottoms                   
            mean_val_bot_l = mean(bot_l); % Mean values among bottoms 
            no_of_bot_l = length(bot_l);  % Number of bottoms
            
            % For right foot force
            [bot_r, bot_locs_r, bot_w_r, bot_p_r] = findpeaks(-B(:, 2),Fs,'MinPeakDistance',1.16);
            mean_dist_bot_r = mean(diff(bot_locs_r));
            mean_val_bot_r = mean(bot_r);
            no_of_bot_r = length(bot_r);
            
            % RMS of bottoms
            RMS_bot_l = rms(bot_l);
            RMS_bot_r = rms(bot_r);
            
            % Absolute change of mean peak and mean bottom
            Change_l = abs(mean_val_pks_l - mean_val_bot_l);
            Change_r = abs(mean_val_pks_r - mean_val_bot_r);
            
            %-------------- Frequency domain using FFT ----------------%
            FFT_l = real(fft(B(:, 1)));
            FFT_r = real(fft(B(:, 2)));
            
            % RMS in FFT
            RMS_FFT_l = rms(FFT_l);
            RMS_FFT_r = rms(FFT_r);
            
            % Variance in FFT
            Var_FFT_l = var(FFT_l);
            Var_FFT_r = var(FFT_r); 
            
            % Fundamental frequency estimation from spectrual peaks
            Max_FFT_l = max(FFT_l);
            Max_FFT_r = max(FFT_r);
            
            % Minimum value in FFT
            Min_FFT_l = min(FFT_l);
            Min_FFT_r = min(FFT_r);
            
            
            %------------ Power Spectral Density (PSD) Estimates ----------%
            Fs = 300;             % Sampling frequency    
            t = 0:1/Fs:1-1/Fs;    % Time vector
            
            N = length(B(:, 1));  % Length of signal
            xdft_l = fft(B(:, 1));
            xdft_l = xdft_l(1:N/2+1);
            xdft_r = fft(B(:, 2));
            xdft_r = xdft_r(1:N/2+1);
            
            % Calculate specturm and get frequency estimate (spectral peak)
            % For left foot force
            psd_l = (1/(Fs*N)) * abs(xdft_l).^2;
            psd_l(2:end-1) = 2*psd_l(2:end-1);
            freq_l = 0:Fs/length(B(:, 1)):Fs/2;
            Max_powerL = max(psd_l);
            
            % For right foot force
            psd_r = (1/(Fs*N)) * abs(xdft_r).^2;
            psd_r(2:end-1) = 2*psd_r(2:end-1);
            freq_r = 0:Fs/length(B(:, 2)):Fs/2;
            Max_powerR = max(psd_r);
            
            % RMS in PSD
            RMS_PSD_l = rms(psd_l);
            RMS_PSD_r = rms(psd_r);
            
            % Variance in PSD
            Var_PSD_l = var(psd_l);
            Var_PSD_r = var(psd_r); 

            %---------- Autocorrelation function ---------------
%             p=100;
%             ACF_l= acf(B(:,1),p);
%             ACF_r= acf(B(:,2),p);          
%             zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);
%             zl = zci(ACF_l);
%             zl= zl(1,1);
%             zr = zci(ACF_r);
%             zr=zr(1,1);
            
            %---------- Add additional info from "train.txt" ---------%
            names = string(Type)+string(i);

            for k = 1:44
                if names == train(k, 1)
                    Age = train(k, 2);
                    Height = train(k, 3);
                    Weight = train(k, 4);
                    Gender = train(k, 5);
                    Speed =  train(k, 6);
                else
                end
            end

            %------------------ For two-class problem  ---------------%   
            % [H = 1, D = 2]
            
            if j == 1
                RES1 = 1; % j = 1 means it is classifed as Healthy
            else
                RES1 = 2; % j = 2 means it is classifed as Disease
            end
            
            %------------------ For four-class problem  --------------%
            % [H = 1, ALS = 2, HUNT = 3, PARK = 4]
            
            RES2 = j;
            
            VEC = [VEC; RMS_l,RMS_r,Std_dev_l,Std_dev_r,COR,...
                mean_dist_pks_l,mean_val_pks_l,no_of_pks_l,...
                mean_dist_pks_r,mean_val_pks_r,no_of_pks_r,RMS_pks_l,...
                RMS_pks_r,mean_dist_bot_l,mean_val_bot_l,no_of_bot_l,...
                mean_dist_bot_r,mean_val_bot_r,no_of_bot_r,RMS_bot_l,...
                RMS_bot_r,Change_l,Change_r,RMS_FFT_l,RMS_FFT_r,...
                Var_FFT_l,Var_FFT_r,Max_FFT_l,Max_FFT_r,Min_FFT_l,...
                Min_FFT_r,Max_powerL,Max_powerR,RMS_PSD_l,RMS_PSD_r,...
                Age,Height,Weight,Gender,Speed,RES1,RES2];
        end
    end
end

DATA = VEC;
cell2csv('FinalFeatures.csv', DATA)


      
