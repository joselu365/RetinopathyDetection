**Data records

This dataset is publicly available at https://www.aiforeye.cn/ and https://figshare.com/s/792d1b02f65be0c08214, 
which is stored as a zip file. In the unzipped folder, all the raw fundus images, the exudate annotations, the DR 
grading labels, and the OD and fovea location annotations are stored in three subfolders, namely "originalImages", 
"exudateLabels", and "odFoveaLabels". In the "originalImages" folder, files are saved in the JPG format and named 
as "n.jpg", with n ranging between 0001 and 1219 indicating the n^th sample. In that folder, we also 
provide a comma-separated-values (CSV) file named "drLabels.csv", wherein the first column indicates the file 
name, the second column indicates the left-versus-right eye categories with 0 representing left eyes and 1 right 
eyes, the third column indicates the DR category assessed via the International Clinical DR Severity Scale (0 to 5, 
with 0 representing normal healthy, and 1 to 5 respectively representing mild non-proliferative DR, moderate 
non-proliferative DR, severe non-proliferative DR, proliferative DR, and DR with laser spots or scars), the fourth 
column indicates the DR grade assessed via the American Academy of Ophthalmology protocol, and the fifth 
column indicates the DR grade assessed via the Scottish DR grading protocol. Another CSV file named 
"c5_DR_reclassified.csv" provides the DR labels for images belonging to category 5 assessed via the three 
protocols. The exudate detection labels, OD bounding box's coordinates, as well as fovea location's coordinates
 are saved in the XML format stored at the corresponding folders (namely "exudateLabels" and "odFoveaLabels"), 
following the same specifications as the Pascal Voc dataset [1]. Hard and soft exudates 
are labeled separately in this dataset. In the XML files, "ex" stands for hard exudates and "se" for soft exudates.

**Usage Notes

After copying all images from the "originalImages" folder to the "exudateLabels" and "odFoveaLabels" folders, 
users can directly open the provided fundus images and the corresponding exudate detection labels, OD bounding 
box's coordinates, as well as fovea location's coordinates using Labelimg (a graphical image annotation tool, which 
can be accessed at https://github.com/tzutalin/labelImg). This tool provides functions of visualizations and 
modifications of annotations (according to research needs). Please note in order to display directly in Labelimg, 
the fovea location's coordinates are transformed into a small box (F_x,F_y,F_{x+1},F_{y+1}).

[1] Everingham, M., Eslami, S. A., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A. (2015). 
The pascal visual object classes challenge: A retrospective. International journal of computer vision, 111(1), 98-136.