

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b e)
(ontable c)
(on d f)
(on e c)
(on f b)
(clear a)
(clear d)
)
(:goal
(and
(on a f)
(on b e)
(on d a)
(on f b))
)
)


