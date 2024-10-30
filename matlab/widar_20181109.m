a=[1,2,3];
for y=1:length(a) % Volunteer Status
    for b=1:6  % Different Action Categories
        for t=1:5
            for f=1:5
                for r=1:10
                    csi_trace=read_bf_file(['E:\exper_datasave\csi data\20181109\user' num2str(a(y)) '\user' num2str(a(y)) '-' num2str(b) '-' num2str(t) '-' num2str(f) '-' num2str(r) '-r6.dat']);  
                    csi_trace(cellfun(@isempty,csi_trace))=[];
                    [H,L]=size(csi_trace); % The function assigns the number of rows to `H` and the number of columns to `L`.
                    if H < 1344 
                        continue 
                    end
                    H = 1344; 
                    for i=1:H
                        csi_entry=csi_trace{i};
                        csi1=get_scaled_csi(csi_entry);  % Extracting the CSI matrix
                        csi2=csi1(1,:,:);   %Retrieve the first row from all pages. 
                        csi_a=squeeze(csi2);  % Dimensionality reduction.
                        csi_b=csi_a.';  
                        csi=abs(csi_b); % Absolute value for real numbers / magnitude for complex numbers.
                        
                        
                        ant1_csi(i,:)=csi(:,1);  
                        ant2_csi(i,:)=csi(:,2);
                        ant3_csi(i,:)=csi(:,3);
                    end
                    combined_csi = [ant1_csi, ant2_csi, ant3_csi];
                    Hd = MyButterworth;
                    traindata=filter(Hd,abs(combined_csi)); 
                    traindata=traindata.';
           
                    a1='E:\exper_datasave\csi data\20181109_3ants\';
                    a2=b;  
                    a3='-';
                    a4=a(y);  
                    a5='-';
                    a6=t;
                    a7='-';
                    a8=f;
                    a9='-';
                    a10=r;
                    a11='-r3.mat';
                    str=sprintf('%s%d%s%d%s%d%s%d%s%d%s',a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11);
                    save(str,'traindata');
                    
                end
            end
        end
    end
end


