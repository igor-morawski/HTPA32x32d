#PS script to convert all *.txt in folder to .np arrays (not normalized) and, gifs
#using conda
$dir = " ";
$files = Get-Childitem -Path $dir -Filter *.txt;
$rot = 2
foreach ($file in $files) {
    $name = [io.path]::GetFileNameWithoutExtension($file);
    $parent = $file.Directory.FullName;
    $in = "$($file.FullName)";
    $out = "$parent\$name.npy";
    $gif = "$parent\$name.gif";
    $cmd = "python txt2np.py --input=$in --output=$out --gif=$gif --rot90=$rot";
    Invoke-Expression -Command $cmd;
    }