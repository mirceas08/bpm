function writesparse(matrix, filename)
    [i, j, v_ij] = find(matrix);
    data_dump = [i, j, v_ij];

    fid = fopen(filename, 'w');
    fprintf(fid, '%d %d %.2f\r\n', data_dump');
    fclose(fid);
