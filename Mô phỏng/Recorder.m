%% Chay section nay 1 lan dau tien (Nho thay doi dia chi duong dan luu file)
Fs    =    16E+3;
nBits    =    24;  
nChannels = 1; 

cd 'E:\Documents\xxxxxx\New folder\Temp'

command = audiorecorder(Fs, nBits, nChannels); 
command.StartFcn = 'disp(''Start speaking.'')';
command.StopFcn = 'disp(''End of recording.'')';
i = 1000;

%% Chay section nay lien tuc de ghi am
clc
recordblocking(command,1); 
commandsound = getaudiodata(command);

nowrecording = 'Long_dung';                         %Thay doi ten khi ghi am lenh moi (tien, lui, trai, phai, dung)
filename = [nowrecording,'_', num2str(i), '.wav'];   
audiowrite(filename, commandsound, Fs)
i = i + 1;