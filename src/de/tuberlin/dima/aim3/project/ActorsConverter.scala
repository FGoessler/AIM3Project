package de.tuberlin.dima.aim3.project

import scala.io.Source


object ActorsConverter {
  def main(args: Array[String]): Unit = {
    val lines = Source.fromFile(new java.io.File("/data/home/fgoessler/actors.list"), "ISO-8859-1").getLines()
    val out = new java.io.PrintWriter(new java.io.File("/data/home/fgoessler/actors-converted.list"))

    var curActor = ""
    for (l <- lines) {
      val splits = l.split("(\\s\\s+)|\\t+")
      if (splits.nonEmpty) {
        if (!splits(0).isEmpty) {
          curActor = splits(0)
        }
        if (splits.length > 1) {
          out.write(curActor + "\t" + splits(1) + "\n")
        }
      }
    }

    out.close()
  }
}
