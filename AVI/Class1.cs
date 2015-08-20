using AForge.Video.FFMPEG;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;


namespace AVI
{
    static public class Writer
    {
        static VideoFileWriter writer;
        static public void  NewWriter(string filename, int w, int h)
        {
            writer = new VideoFileWriter();
            // create new AVI file and open it
            writer.Open(filename, w, h,30, VideoCodec.Default , 5500000);
            
        }

        static public void Close()
        {
            writer.Close();
        }
        static public void AddFrame(Bitmap frame)
       {
           writer.WriteVideoFrame(frame);
       }

       
    }
}
