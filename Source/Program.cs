using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;
using System.Threading;
using System.Drawing;
using System.Diagnostics;
using System.Threading.Tasks;

namespace EncogExample
{
    public class Program
    {

      

        public class Life
        {
            double pL, pR, pB, pT = 0.0; //значения движителей
            double sL, sR, sB, sT = 0.0; //значения сенсоров
            public int nearcount=0; //число, количество сближений данного объекта с другими (аналог, учавствовал в бою, выжил, стал сильней) - показатель силы
            public int borncount = 0; //число съеденных данным объектом - показатель силы
            public double x { get; set; } 
            public double y { get; set; }
            public string id = Guid.NewGuid().ToString();
            BasicNetwork network = new BasicNetwork();
            Random rnd = new Random((int)(DateTime.Now.Ticks%int.MaxValue ) );
            List<Life> World; 
            public Life(List<Life> world,double X,double Y)
            {
                x = X;
                y = Y;
                pL = pR = pB = pT = 0.0; 
                sL = sR = sB = sT = 0.0;
                network.AddLayer(new BasicLayer(null, true, 4)); //создание простой многослойной нейронной сети
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, rnd.Next(100) + 10));
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 4));
                network.Structure.FinalizeStructure();
                network.Reset();
                World = world;
            }
            List<double[]> Memory = new List<double[]>();
            List<double[]> MemorySense = new List<double[]>();

            void Train()
            {
                if (Memory.Count>0)
                {
                    network.Reset();
                    double[][] InputData = new double[Memory.Count][]; //подготовка данных для обучения сети
                    double[][] SenseData = new double[Memory.Count][];
                    for (int i = 0; i < Memory.Count; i++)
                    {
                        InputData[i] = Memory[i];
                        SenseData[i] = MemorySense[i];
                    }
                    IMLDataSet trainingSet = new BasicMLDataSet(InputData, SenseData); 
                    IMLTrain train = new ResilientPropagation(network, trainingSet);

                    int epoch = 1;

                    double old = 9999;
                    double d = 999;
                    do
                    {
                        train.Iteration();
                        //Console.SetCursorPosition(0, 0); //вывод информации о текущем состоянии обучения
                        //Console.Write(@"Epoch #" + epoch + @" Error:" + train.Error);
                        epoch++;
                        d = Math.Abs(old - train.Error);
                        old = train.Error;
                    } while (train.Error > 0.0001 && epoch < 3000 && d > 0.00001);

                    train.FinishTraining();

                    //double sumd=0.0; //подсчет суммарной ошибки после обучения
                    //foreach (IMLDataPair pair in trainingSet) 
                    //{
                    //    IMLData output = network.Compute(pair.Input);
                    //    sumd = sumd + Math.Abs(pair.Ideal[0] - output[0]);
                    //    sumd = sumd / trainingSet.InputSize;
                    //}
                }
            }

            void SaveToMemory()
            {
                double[] Data = { sL, sR, sB, sT };
                double[] SenseData = { pL, pR, pB, pT };
                Memory.Add(Data);
                MemorySense.Add(SenseData);
                if (Memory.Count > 1000)
                {
                    if (nearcount==0)
                    {
                        World.Remove(this);
                        Life newlife = new Life(World, x, y);
                        World.Add(newlife);
                    }
                    Memory.RemoveAt(0);
                    MemorySense.RemoveAt(0);
                }
                //Console.SetCursorPosition(0, 3);
                //Console.Write(Memory.Count.ToString());
            }

            int step = 0;
            int stepmax = 0;
            public void DoLive()
            {
                RefreshSense(); //look around
                Move();//do step
                step++;
                if (step>stepmax )
                {
                    int maxstep = 5-(int)(sL + sR + sB + sT);
                    stepmax = rnd.Next(40 / (maxstep));
                    step = 0;
                    SaveToMemory();
                    Train();
                }
                double[][] Input = { new double[] { 0, 0, 0, 0 } };
                IMLDataSet trainingSet;
                //thinking
                //если я тот кто ближе слабее попробовать съесть, если нет драпать
                if (nearlife.borncount > borncount )
                {
                    double[][] SenseData = { new double[] { 0, 0, 0, 0 } };
                    trainingSet = new BasicMLDataSet(Input, SenseData);
                }
                else
                {
                    double[][] SenseData = { new double[] { 1, 1, 1, 1 } };
                    trainingSet = new BasicMLDataSet(Input, SenseData);
                }
                IMLData output = network.Compute(trainingSet[0].Ideal   );
                if (output[0] > pL) { pL += 0.001; } else { pL -= 0.001; }
                if (output[1] > pR) { pR += 0.001; } else { pR -= 0.001; }
                if (output[2] > pB) { pB += 0.001; } else { pB -= 0.001; }
                if (output[3] > pT) { pT += 0.001; } else { pT -= 0.001; }
               
            }

            

            void Move()
            {
                if (pL>0.01)
                {
                    pL = 0.01;
                }
                else if (pL < 0)
                {
                    pL = 0;
                }
                if (pR > 0.01)
                {
                    pR = 0.01;
                }
                else if (pR < 0)
                {
                    pR = 0;
                }
                if (pB > 0.01)
                {
                    pB = 0.01;
                }
                else if (pB < 0)
                {
                    pB = 0;
                }
                if (pT > 0.01)
                {
                    pT = 0.01;
                }
                else if (pT < 0)
                {
                    pT = 0;
                }
                x += pL - pR;
                y += pB - pT;
                if (x < 0)
                {
                    x = 1 - x;
                }
                if (y < 0)
                {
                    y = 1 - y;
                }
                if (x > 1)
                {
                    x = x - 1;
                }
                if (y > 1)
                {
                    y = y - 1;
                }
            }
            Life nearlife = null;
            void RefreshSense()
            {
               ret1: double mind = 99999;

               try
               {
                   foreach (Life life in World)
                   {
                       if (life.id != this.id)
                       {
                           var d = GetDistLife(life);
                           if (d < mind)
                           {
                               mind = d;
                               nearlife = life;
                               if (d < 0.008)
                               {
                                   if (rnd.NextDouble() > 0.5)//born new life form 60%
                                   {
                                       //Life newlife = new Life(World, x , y );
                                       //world.Add(newlife);
                                       borncount += 1;
                                       if (borncount > 5)
                                       {
                                           borncount = 5;
                                       }
                                       nearlife.borncount += 1;
                                       if (nearlife.borncount > 5)
                                       {
                                           nearlife.borncount = 5;
                                       }
                                       if (nearlife.nearcount < nearcount)
                                       {
                                           World.Remove(nearlife);
                                       }
                                       else
                                       {
                                           World.Remove(this);
                                       }
                                       //World.Remove(nearlife);
                                       //World.Remove(this);
                                       goto ret1;
                                   }
                                   nearcount += 1;
                                   if (nearcount > 5)
                                   {
                                       nearcount = 5;
                                   }
                                   nearlife.nearcount += 1;
                                   if (nearlife.nearcount > 5)
                                   {
                                       nearlife.nearcount = 5;
                                   }
                                   //x = (rnd.NextDouble()+x)/2;
                                   //y = (rnd.NextDouble()+y)/2;
                                   //life.x = (rnd.NextDouble() + 0.04 + life.x) / 2;
                                   //life.y = (rnd.NextDouble() + 0.04 + life.y) / 2;
                               }
                           }
                       }
                   }
               }
               catch (Exception)
               {
                   goto ret1;
               }
                
                if (nearlife != null)
                {
                    sL = GetDist(x - 0.05, nearlife.x, y, nearlife.y);
                    sR = GetDist(x + 0.05, nearlife.x, y, nearlife.y);
                    sB = GetDist(x, nearlife.x, y - 0.05, nearlife.y);
                    sT = GetDist(x, nearlife.x, y + 0.05, nearlife.y);
                }
            }

            double GetDistLife(Life l1)
            {
                return GetDist(l1.x, x, l1.y, y);
            }

            double GetDist(double x1, double x2, double y1, double y2)
            {
                return Math.Sqrt(Math.Pow(x1 - x2, 2) + Math.Pow(y1 - y2, 2));
            }
        }

        static double GetDist(double x1, double x2, double y1, double y2)
        {
            return Math.Sqrt(Math.Pow(x1 - x2, 2) + Math.Pow(y1 - y2, 2));
        }

        static List<Life> world = new List<Life>();
        static Bitmap frame;
       
        static void Draw()
        {
            //Console.Clear();
            //g.Clear(Color.Black);
            g.FillRectangle(new SolidBrush(Color.FromArgb(25, Color.Black)), new Rectangle(0, 0, frame.Width, frame.Height));
            foreach (var life in world)
            {

                int xx = (int)Math.Ceiling(life.x * 50.0);
                int yy = (int)Math.Ceiling(life.y * 40.0);

                Console.SetCursorPosition(xx, yy);
                if (life.borncount > 0)
                {
                    Console.ForegroundColor = ConsoleColor.Magenta;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.White;
                }
                Console.Write("#");
                Life nearlife =null ; double mind = 9999;
                xx = (int)Math.Ceiling(life.x * frame.Width );
                yy = (int)Math.Ceiling(life.y * frame.Height );
                
                foreach (Life lifeform in world)
                {
                    if (life.id != lifeform.id)
                    {
                        var d = GetDist(lifeform.x ,life.x,lifeform.y,life.y );
                        if (d < mind)
                        {
                            mind = d;
                            nearlife = lifeform;
                            if (d < 0.02)
                            {
                             g.FillEllipse(new SolidBrush(Color.FromArgb(255, 255, 255,255)), new Rectangle(xx - 15, yy - 15, 30, 30));
                            }
                        }
                    }
                }
                byte dk = (byte)(life.borncount * 45);
                byte dg = (byte)(life.nearcount * 45);
                if (nearlife != null)
                {
                    var sL = GetDist(life.x - 0.05, nearlife.x, life.y, nearlife.y);
                    var sR = GetDist(life.x + 0.05, nearlife.x, life.y, nearlife.y);
                    var sB = GetDist(life.x, nearlife.x, life.y - 0.05, nearlife.y);
                    var sT = GetDist(life.x, nearlife.x, life.y + 0.05, nearlife.y);
                    byte dL=(byte)Math.Floor(Math.Abs(255-sL * 400.0));
                    byte dR=(byte)Math.Floor(Math.Abs(255-sR * 400.0));
                    byte dB=(byte)Math.Floor(Math.Abs(255-sB * 400.0));
                    byte dT=(byte)Math.Floor(Math.Abs(255-sT * 400.0));

                    g.FillEllipse(new SolidBrush(Color.FromArgb(255, dL, dk, dg)), new Rectangle(xx - 15, yy - 5, 5, 5));
                    g.FillEllipse(new SolidBrush(Color.FromArgb(255, dR, dk, dg)), new Rectangle(xx + 5, yy - 5, 5, 5));
                    g.FillEllipse(new SolidBrush(Color.FromArgb(255, dB, dk, dg)), new Rectangle(xx - 5, yy + 5, 5, 5));
                    g.FillEllipse(new SolidBrush(Color.FromArgb(255, dT, dk, dg)), new Rectangle(xx - 5, yy - 15, 5, 5));
                }
                g.FillEllipse(new SolidBrush(Color.FromArgb(255, dk, dg)), new Rectangle(xx - 10, yy - 10, 15, 15));
            }
        }

        static Graphics g;
        static void Main(string[] args)
        {
            // create a neural network, without using a factory
            Random rnd = new Random((int)(DateTime.Now.Ticks % int.MaxValue));

            for (int i = 0; i < 35; i++)
            {
                Life life = new Life(world, rnd.NextDouble(), rnd.NextDouble());
                world.Add(life);
            }
          
            int width = 1280;
            int height = 1024;
            frame = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            string name = String.Format("{0}.avi", DateTime.Now.ToString());
            String outputFilePath = name.Replace(":", "_") ;
            AVI.Writer.NewWriter(outputFilePath, width, height);
            //AnimatedGifEncoder e = new AnimatedGifEncoder();
            //e.Start(outputFilePath);
            //e.SetDelay(5);
            //e.SetRepeat(0);
            //add a new video stream and one frame to the new file

            //var writer = new VideoWriter(Guid.NewGuid().ToString() + ".avi", new DotImaging.Primitives2D.Size(640, 480), 30f, true);
            //writer.Open();

            int num = 0;
            g = Graphics.FromImage(frame);
            do
            {
                num++;
                try
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();
                    //foreach (var life in world)
                    //{
                    //    try
                    //    {
                    //        life.DoLive();
                    //    }
                    //    catch (Exception)
                    //    {
                    //    }
                    //}
                    Parallel.ForEach(world.Cast<Life>(),
                    life =>
                    {
                        life.DoLive();
                    });
                    sw.Stop();
                    //Console.SetCursorPosition(0, 4);
                    //Console.Write("Tick: " + sw.ElapsedMilliseconds);
                    sw.Reset(); sw.Start();
                    Draw();
                    
                    //frame.Save(name);
                    //if (num%2==0)
                    //{
                    //    e.AddFrame(frame);
                    //}
                    AVI.Writer.AddFrame(frame);
                    sw.Stop();
                    //Console.SetCursorPosition(0, 5);
                    //Console.Write("Draw: " + sw.ElapsedMilliseconds);
                    
                }
                catch (Exception esx)
                {
                    //Console.SetCursorPosition(0, 6);
                    //Console.Write(esx.Message + esx.Source + esx.StackTrace);
                }
                //Console.SetCursorPosition(0, 7);
                //Console.Write(num);
                if (num>=100)
                {
                    num = 0;
                    Console.Clear();
                    //AVI.Writer.Close();
                    ////e.Finish();
                    //break;
                }
            } while (true);
            EncogFramework.Instance.Shutdown();
        }
    }
}