#pragma once
#ifndef _FILE_OUTPUT_H_
#define _FILE_OUTPUT_H_




class File_output
{
    public:
    File_output();
    ~File_output();

    void file_write(float4 *d_pos,uint totalVerts, const char* filename);

};


#endif // // _FILE_OUTPUT_H_